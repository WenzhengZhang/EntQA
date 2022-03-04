import torch
import json
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import faiss
from utils import sample_range_excluding
import random
from preprocess_data import normalize_string


# for embedding entities during inference
class EntitySet(Dataset):
    def __init__(self, entities):
        self.entities = entities

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        entity = self.entities[index]
        entity_token_ids = torch.tensor(entity['text_ids']).long()
        entity_masks = torch.tensor(entity['text_masks']).long()
        return entity_token_ids, entity_masks


# For embedding all the mentions during inference
class MentionSet(Dataset):
    def __init__(self, mentions, max_len, tokenizer,
                 add_topic=True, use_title=False
                 ):
        self.mentions = mentions
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.add_topic = add_topic
        self.use_title = use_title
        # [2] is token id of '[unused1]' for bert tokenizer
        self.TT = [2]

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        mention = self.mentions[index]
        if self.add_topic:
            title = mention['title'] if self.use_title else mention['topic']
            title_ids = self.TT + title
        else:
            title_ids = []
        # CLS + mention ids + TT + title ids
        mention_title_ids = mention['text']+title_ids
        mention_ids = (mention_title_ids + [self.tokenizer.pad_token_id] * (
                self.max_len - len(mention_title_ids)))[:self.max_len]
        mention_masks = ([1] * len(mention_title_ids) + [0] * (
                self.max_len - len(mention_title_ids)))[:self.max_len]
        mention_token_ids = torch.tensor(mention_ids).long()
        mention_masks = torch.tensor(mention_masks).long()
        return mention_token_ids, mention_masks


def get_labels(samples, all_entity_map):
    # get labels for samples
    labels = []
    for sample in samples:
        entities = sample['entities']
        label_list = [all_entity_map[normalize_string(e)] for
                      e in
                      entities if e in all_entity_map]
        labels.append(label_list)
    labels = np.array(labels)
    return labels


def get_group_indices(samples):
    # get list of group indices for passages come from the same document
    doc_ids = np.unique([s['doc_id'] for s in samples])
    group_indices = {k: [] for k in doc_ids}
    for i, s in enumerate(samples):
        doc_id = s['doc_id']
        group_indices[doc_id].append(i)
    return list(group_indices.values())


def get_entity_map(entities):
    #  get all entity map: map from entity title to index
    entity_map = {}
    for i, e in enumerate(entities):
        entity_map[e['title']] = i
    assert len(entity_map) == len(entities)
    return entity_map


class RetrievalSet(Dataset):
    def __init__(self, mentions, entities, labels, max_len,
                 tokenizer, candidates,
                 num_cands, rands_ratio, type_loss,
                 add_topic=True, use_title=False):
        self.mentions = mentions
        self.candidates = candidates
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.labels = labels
        self.num_cands = num_cands
        self.rands_ratio = rands_ratio
        self.all_entity_token_ids = np.array([e['text_ids'] for e in entities])
        self.all_entity_masks = np.array([e['text_masks'] for e in entities])
        self.entities = entities
        self.type_loss = type_loss
        self.add_topic = add_topic
        self.use_title = use_title
        # '[unused1]' for bert tokenizer
        self.TT = [2]

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        """

        :param index: The index of mention
        :return: mention_token_ids,mention_masks,entity_token_ids,entity_masks : 1 X L
                entity_hard_token_ids, entity_hard_masks: k X L  (k<=10)
        """
        # process mention
        mention = self.mentions[index]
        if self.add_topic:
            title = mention['title'] if self.use_title else mention['topic']
            title_ids = self.TT + title
        else:
            title_ids = []
        # CLS + mention ids + TT + title ids
        mention_title_ids = mention['text'] + title_ids
        mention_ids = mention_title_ids + [self.tokenizer.pad_token_id] * (
                self.max_len - len(mention_title_ids))
        mention_masks = [1] * len(mention_title_ids) + [0] * (
                self.max_len - len(mention_title_ids))
        mention_token_ids = torch.tensor(mention_ids[:self.max_len]).long()
        mention_masks = torch.tensor(mention_masks[:self.max_len]).long()
        # process entity
        cand_ids = []
        labels = self.labels[index]
        # dummy labels if there is no label entity for the given passage
        if len(labels) == 0:
            labels = [-1]
        else:
            labels = list(set(labels))
        cand_ids += labels
        num_pos = len(labels)
        # assert num_pos >= 0
        num_neg = self.num_cands - num_pos
        assert num_neg >= 0
        num_rands = int(self.rands_ratio * num_neg)
        num_hards = num_neg - num_rands
        # non-hard and non-label for random negatives
        rand_cands = sample_range_excluding(len(self.entities), num_rands,
                                            set(labels).union(set(
                                                self.candidates[index])))
        cand_ids += rand_cands
        # process hard negatives
        if self.candidates is not None:
            # hard negatives
            hard_negs = random.sample(list(set(self.candidates[index]) - set(
                labels)), num_hards)
            cand_ids += hard_negs
        passage_labels = torch.tensor([1] * num_pos + [0] * num_neg).long()
        candidate_token_ids = self.all_entity_token_ids[cand_ids].tolist()
        candidate_masks = self.all_entity_masks[cand_ids].tolist()
        assert passage_labels.size(0) == self.num_cands
        candidate_token_ids = torch.tensor(candidate_token_ids).long()
        assert candidate_token_ids.size(0) == self.num_cands
        candidate_masks = torch.tensor(candidate_masks).long()
        return mention_token_ids, mention_masks, candidate_token_ids, \
               candidate_masks, passage_labels


def load_data(data_dir, kb_dir):
    """

    :param data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data')

    def load_mentions(part):
        with open(os.path.join(data_dir, 'tokenized_aida_%s.json' % part)) as f:
            mentions = json.load(f)
        return mentions

    samples_train = load_mentions('train')
    samples_val = load_mentions('val')
    samples_test = load_mentions('test')

    def load_entities():
        entities = []
        with open(os.path.join(kb_dir, 'entities_kilt.json')) as f:
            for line in f:
                entities.append(json.loads(line))

        return entities

    entities = load_entities()

    return samples_train, samples_val, samples_test, entities


def get_embeddings(loader, model, is_mention, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks = batch
            k1, k2 = ('mention_token_ids', 'mention_masks') if is_mention else \
                ('entity_token_ids', 'entity_masks')
            kwargs = {k1: input_ids, k2: input_masks}
            j = 0 if is_mention else 2
            embed = model(**kwargs)[j].detach()
            embeddings.append(embed.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    model.train()
    return embeddings


def get_hard_negative(mention_embeddings, all_entity_embeds, k,
                      max_num_postives,
                      use_gpu_index=False):
    index = faiss.IndexFlatIP(all_entity_embeds.shape[1])
    if use_gpu_index:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(all_entity_embeds)
    scores, hard_indices = index.search(mention_embeddings,
                                        k + max_num_postives)
    del mention_embeddings
    del index
    return hard_indices, scores


def make_single_loader(data_set, bsz, shuffle):
    loader = DataLoader(data_set, bsz, shuffle=shuffle)
    return loader


def get_loader_from_candidates(samples, entities, labels, max_len,
                               tokenizer, candidates,
                               num_cands, rands_ratio, type_loss,
                               add_topic, use_title, shuffle, bsz
                               ):
    data_set = RetrievalSet(samples, entities, labels,
                            max_len, tokenizer, candidates,
                            num_cands, rands_ratio, type_loss, add_topic,
                            use_title)
    loader = make_single_loader(data_set, bsz, shuffle)
    return loader


def get_loaders(samples_train, samples_val, samples_test, entities, max_len,
                tokenizer, mention_bsz, entity_bsz, add_topic,
                use_title):
    #  get all mention and entity dataloaders
    train_mention_set = MentionSet(samples_train, max_len, tokenizer,
                                   add_topic, use_title)
    val_mention_set = MentionSet(samples_val, max_len, tokenizer, add_topic,
                                 use_title)
    test_mention_set = MentionSet(samples_test, max_len, tokenizer, add_topic,
                                  use_title)
    entity_set = EntitySet(entities)
    entity_loader = make_single_loader(entity_set, entity_bsz, False)
    train_men_loader = make_single_loader(train_mention_set, mention_bsz,
                                          False)
    val_men_loader = make_single_loader(val_mention_set, mention_bsz, False)
    test_men_loader = make_single_loader(test_mention_set, mention_bsz, False)

    return train_men_loader, val_men_loader, test_men_loader, entity_loader


def save_candidates(mentions, candidates, entity_map, labels, out_dir, part):
    # save results for reader training
    assert len(mentions) == len(candidates)
    labels = labels.tolist()
    out_path = os.path.join(out_dir, '%s.json' % part)
    entity_titles = np.array(list(entity_map.keys()))
    fout = open(out_path, 'w')
    for i in range(len(mentions)):
        mention = mentions[i]
        m_candidates = candidates[i].tolist()
        m_spans = [[s[0], s[1] - 1] for s in mention['spans']]
        assert len(mention['entities']) == len(mention['spans'])
        ent_span_dict = {k: [] for k in mention['entities']}
        for j, l in enumerate(mention['entities']):
            ent_span_dict[l].append(m_spans[j])
        if part == 'train':
            positives = [c for c in m_candidates if c in labels[i]]
            negatives = [c for c in m_candidates if c not in labels[i]]
            pos_titles = entity_titles[positives].tolist()
            pos_spans = [ent_span_dict[p] for p in pos_titles]
            gold_ids = list(set(labels[i]))
            gold_titles = entity_titles[gold_ids].tolist()
            gold_spans = [ent_span_dict[g] for g in gold_titles]
            neg_spans = [[[0, 0]]] * len(negatives)
            item = {'doc_id': mention['doc_id'],
                    'mention_idx': i,
                    'mention_ids': mention['text'],
                    'positives': positives,
                    'negatives': negatives,
                    'labels': mention['entities'],
                    'label_spans': m_spans,
                    'gold_ids': gold_ids,
                    'gold_spans': gold_spans,
                    'pos_spans': pos_spans,
                    'neg_spans': neg_spans,
                    'offset': mention['offset'],
                    'title': mention['title'],
                    'topic': mention['topic'],
                    'passage_labels': [1] * len(positives) + [0] * len(
                        negatives)
                    }
        else:
            candidate_titles = entity_titles[m_candidates]
            candidate_spans = [ent_span_dict[s] if s in ent_span_dict else
                               [[0, 0]] for s in candidate_titles]
            passage_labels = [1 if c in mention['entities'] else 0 for c in
                              candidate_titles]
            item = {'doc_id': mention['doc_id'],
                    'mention_idx': i,
                    'candidates': m_candidates,
                    'title': mention['title'],
                    'topic': mention['topic'],
                    'mention_ids': mention['text'],
                    'labels': mention['entities'],
                    'label_spans': m_spans,
                    'label_ids': labels[i],
                    'offset': mention['offset'],
                    'candidate_spans': candidate_spans,
                    'passage_labels': passage_labels
                    }
        fout.write('%s\n' % json.dumps(item))
    fout.close()
