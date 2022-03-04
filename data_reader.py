import torch
import json
from torch.utils.data import Dataset
import os
import numpy as np
from utils import OrderedSet
from data_retriever import make_single_loader


class ReaderData(Dataset):
    # get the input data item for the reader model
    def __init__(self,
                 tokenizer,
                 samples,
                 entities,
                 max_len,
                 max_num_candidates,
                 is_training,
                 add_topic=False,
                 use_title=False):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.samples = samples
        self.entities = entities
        self.all_entity_token_ids = np.array([e['text_ids'] for e in entities])
        self.all_entity_masks = np.array([e['text_masks'] for e in entities])
        self.max_len = max_len
        self.max_num_candidates = max_num_candidates
        self.add_topic = add_topic
        self.use_title = use_title
        self.TT = [2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        title = None
        if self.add_topic:
            title = sample['title'] if self.use_title else sample['topic']
        mention_ids = sample['mention_ids']
        passage_labels = sample['passage_labels'][:self.max_num_candidates]
        if self.add_topic:
            title_ids = self.TT + title
        else:
            title_ids = []
        if self.is_training:
            positives = sample['positives']
            pos_spans = sample['pos_spans']
            assert len(positives) == len(pos_spans)
            # ensure always have positive labels for training
            if len(positives) == 0:
                positives = sample['gold_ids']
                pos_spans = sample['gold_spans']
                passage_labels = ([1] * len(positives) + passage_labels)[
                                 :self.max_num_candidates]
            negatives = list(np.random.permutation(sample['negatives']))
            candidates = (positives + negatives)[:self.max_num_candidates]
            spans = (pos_spans + sample['neg_spans'])[
                    :self.max_num_candidates]
        else:
            candidates = sample['candidates'][:self.max_num_candidates]
            spans = sample['candidate_spans'][:self.max_num_candidates]
        candidates_ids = self.all_entity_token_ids[candidates]
        candidates_masks = self.all_entity_masks[candidates]

        encoded_pairs = torch.zeros((self.max_num_candidates,
                                     self.max_len)).long()
        type_marks = torch.zeros((self.max_num_candidates, self.max_len)).long()
        attention_masks = torch.zeros((self.max_num_candidates,
                                       self.max_len)).long()
        answer_masks = torch.zeros((self.max_num_candidates,
                                    self.max_len)).long()
        passage_labels = torch.tensor(passage_labels).long()
        if self.is_training:
            start_labels = torch.zeros((self.max_num_candidates,
                                        self.max_len)).long()
            end_labels = torch.zeros((self.max_num_candidates,
                                      self.max_len)).long()
        for i, candidate_ids in enumerate(candidates_ids):
            if self.is_training:
                _spans = np.array(spans[i])
                start_labels[i, _spans[:, 0]] = 1
                end_labels[i, _spans[:, 1]] = 1
            candidate_ids = candidate_ids.tolist()
            candidate_masks = candidates_masks[i].tolist()
            # CLS mention ids TT title ids SEP candidate ids SEP
            input_ids = mention_ids[:-1] + title_ids + [
                self.tokenizer.sep_token_id] + candidate_ids[1:]
            input_ids = (input_ids + [self.tokenizer.pad_token_id] * (
                    self.max_len - len(input_ids)))[:self.max_len]
            attention_mask = [1] * (len(mention_ids + title_ids)) + \
                             candidate_masks[1:]
            attention_mask = (attention_mask + [0] * (self.max_len - len(
                attention_mask)))[:self.max_len]
            token_type_ids = [0] * len(mention_ids + title_ids) + \
                             candidate_masks[1:]
            token_type_ids = (token_type_ids + [0] * (self.max_len - len(
                token_type_ids)))[:self.max_len]
            encoded_pairs[i] = torch.tensor(input_ids)
            attention_masks[i] = torch.tensor(attention_mask)
            type_marks[i] = torch.tensor(token_type_ids)
            answer_masks[i, :len(mention_ids)] = 1
        if self.is_training:
            return encoded_pairs, attention_masks, type_marks, answer_masks, \
                   passage_labels, start_labels, end_labels
        else:
            return encoded_pairs, attention_masks, type_marks, answer_masks, \
                   passage_labels


def load_data(data_dir, kb_dir):
    def read_data(part):
        name = '%s.json' % part
        items = []
        with open(os.path.join(data_dir, name)) as f:
            for line in f:
                item = json.loads(line)
                items.append(item)
        return items

    samples_train = read_data('train')
    samples_dev = read_data('val')
    samples_test = read_data('test')

    def load_entities():
        entities = []
        with open(os.path.join(kb_dir, 'entities_kilt.json')) as f:
            for line in f:
                entities.append(json.loads(line))

        return entities

    entities = load_entities()

    return samples_train, samples_dev, samples_test, entities


# get document level gold results
def get_golds(samples_train, samples_dev, samples_test):
    def get_passage_gold(samples):
        p_golds = []
        for sample in samples:
            assert len(sample['labels']) == len(sample['label_spans'])
            # start,end,entity
            g = [span + [entity] for span, entity in zip(sample['label_spans'],
                                                         sample['labels'])]
            p_golds.append(g)
        return p_golds

    p_golds_train = get_passage_gold(samples_train)
    p_golds_val = get_passage_gold(samples_dev)
    p_golds_test = get_passage_gold(samples_test)
    golds_train_doc = get_results_doc(p_golds_train, samples_train)
    golds_val_doc = get_results_doc(p_golds_val, samples_dev)
    golds_test_doc = get_results_doc(p_golds_test, samples_test)
    return golds_train_doc, golds_val_doc, golds_test_doc, p_golds_train, \
           p_golds_val, p_golds_test


def get_loaders(tokenizer, data, max_len,
                max_num_candidates,
                max_num_candidates_val,
                train_bsz, val_bsz,
                add_topic, use_title):
    samples_train, samples_dev, samples_test, entities = data
    train_set = ReaderData(tokenizer, samples_train, entities, max_len,
                           max_num_candidates, True,
                           add_topic, use_title)
    dev_set = ReaderData(tokenizer, samples_dev, entities, max_len,
                         max_num_candidates_val, False, add_topic,
                         use_title)
    test_set = ReaderData(tokenizer, samples_test, entities, max_len,
                          max_num_candidates_val, False, add_topic,
                          use_title)
    loader_train = make_single_loader(train_set, train_bsz, True)
    loader_dev = make_single_loader(dev_set, val_bsz, False)
    loader_test = make_single_loader(test_set, val_bsz, False)
    return loader_train, loader_dev, loader_test


def get_results_doc(passage_results, samples):
    # get document level results from passage-level results
    assert len(passage_results) == len(samples)
    results = []
    # p: start, end, entity_name
    for p, sample in zip(passage_results, samples):
        offset = sample['offset']
        if len(p) == 0:
            continue
        for r in p:
            result = (sample['doc_id'], r[0] + offset, r[1] + offset, r[2])
            results.append(result)
    # result: doc_id, start_doc,end_doc,entity_name
    results = list(OrderedSet(results))
    return results


# save passage level results
def save_results(predicts, p_golds, samples, results_dir, part):
    assert len(predicts) == len(p_golds)
    assert len(samples) == len(predicts)
    save_path = os.path.join(results_dir, 'reader_%s_results.json' % part)
    results = []
    for p_gold, predict, sample in zip(p_golds, predicts, samples):
        result = {}
        result['doc_id'] = sample['doc_id']
        result['text'] = sample['mention_ids']
        result['predicts'] = predict
        result['golds'] = p_gold
        results.append(result)
    with open(save_path, 'w') as f:
        for r in results:
            f.write('%s\n' % json.dumps(r))
