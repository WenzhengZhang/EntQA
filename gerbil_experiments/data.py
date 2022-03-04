import torch
import json
from torch.utils.data import DataLoader, Dataset
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../')))
from data_retriever import MentionSet
from reader import get_predicts
import math


def process_raw_data(document, tokenizer, length, stride):
    topic = document.split(' ', 1)[0].replace(',', '').replace("'s", '')
    title_1 = document.split('.', 1)[0]
    title_2 = document.split('\n', 1)[0]
    title = title_1 if len(title_1) < len(title_2) else title_2
    title = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
    topic = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(topic))
    text = tokenizer.tokenize(document)
    token2char_start, token2char_end = token_to_char_map(document,
                                                         text)
    text_ids = tokenizer.convert_tokens_to_ids(text)
    content_length = length - 2
    samples = []
    data_num = 0
    if len(text_ids) < content_length:
        instance_ids = [101] + text_ids + [102]
        samples.append({'text': instance_ids, 'topic': topic, 'title': title})
    else:
        for ins_num in range(math.ceil(len(text_ids) / stride)):
            begin = ins_num * stride
            end = ins_num * stride + content_length
            instance_ids = [101] + text_ids[begin:end] + [102]
            # +1 for [BOS]
            samples.append(
                {
                    "text": instance_ids,
                    "topic": topic,
                    'title': title
                }
            )

    data_num += 1

    print("finish processing data {}".format(data_num))

    return samples, token2char_start, token2char_end


def load_entities(ents_path):
    entities = []
    with open(ents_path) as f:
        for line in f:
            item = json.loads(line)
            entities.append(item)
    return entities


def get_retriever_loader(samples, tokenizer, bsz, max_len, add_topic=False,
                         use_title=False):
    # get retriever dataloader
    retriever_set = MentionSet(samples, max_len, tokenizer, add_topic,
                               use_title)
    retriever_loader = DataLoader(retriever_set, bsz, shuffle=False)
    return retriever_loader


def get_reader_input(samples, candidates, ents):
    # convert faiss top-k candidates to reader input list of dicts
    # ents is list of entity dicts
    assert len(samples) == len(candidates)
    results = []
    for i, s in enumerate(samples):
        cands = candidates[i]
        result = {'text': s['text'], 'topic': s['topic'],
                  'title': s['title'],
                  'candidates': [ents[j] for j in cands]}
        results.append(result)
    return results


def get_reader_loader(samples, tokenizer, max_len,
                      max_num_candidates, bsz,
                      add_topic, use_title):
    #   get the reader loader
    reader_set = ReaderTestData(tokenizer, samples, max_len,
                                max_num_candidates, add_topic, use_title)
    reader_loader = DataLoader(reader_set, bsz, shuffle=False)
    return reader_loader


def get_raw_results(model, device, loader, k, samples,
                    do_rerank,
                    filter_span=True,
                    no_multi_ents=False):
    model.eval()
    ps = []
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            if do_rerank:
                batch_p = model(*batch)[0]
            else:
                batch_p = model(*batch).detach()
            batch_p = batch_p.cpu()
            ps.append(batch_p)
        ps = torch.cat(ps, 0)
    raw_predicts = get_predicts(ps, k, filter_span, no_multi_ents)
    assert len(raw_predicts) == len(samples)
    return raw_predicts


def normalize_str(s):
    return s.replace(' ', '_')


def process_raw_predicts(raw_predicts, samples):
    print('transforming predicts')
    transformed_predicts = []
    for i, r in enumerate(raw_predicts):
        s = samples[i]
        predict = [p[1:] + [normalize_str(s['candidates'][p[0]]['title'])] for
                   p in r]
        transformed_predicts.append(predict)
    assert len(transformed_predicts) == len(samples)
    return transformed_predicts


def get_doc_level_predicts(predicts, stride):
    results = []
    for i, r in enumerate(predicts):
        previous_len = i * stride
        # -1 because of [101] token

        result = [(t[0] - 1 + previous_len, t[1] - 1 + previous_len,
                   t[2]) if t[0] > 0 else (t[0] + previous_len, t[1] - 1 +
                                           previous_len, t[2]) for t in r]
        results.extend(result)
    results = list(set(results))
    return results


def token_to_char_map(document, doc_tokens):
    char_space_dict = compute_white_after_char(document)
    # doc_tokens = tokenizer.tokenize(document)
    # start index map and end index map
    token2char_start = {}
    token2char_end = {}
    cum_len = 0
    char_index = 0
    for i, token in enumerate(doc_tokens):
        token2char_start[i] = cum_len
        token = token.replace('##', '')
        cum_len += len(token)
        token2char_end[i] = cum_len - 1
        len_token = 1 if token == '[UNK]' else len(token)
        for _ in range(len_token):
            cum_len += char_space_dict[char_index]
            char_index += 1
    return token2char_start, token2char_end


def token_span_to_gerbil_span(predicts, token2char_start, token2char_end):
    results = [(token2char_start[p[0]],
                token2char_end[p[1]] - token2char_start[p[0]] + 1,
                p[2]) for p in predicts]

    return results


# compute whether there is an empty space after a non-empty-space char
def compute_white_after_char(data):
    white_after_char = {}
    whites = [' ', '\n']
    index = 0
    for i, char in enumerate(data):
        if char not in whites:
            num_whites = 0
            for j in range(i + 1, len(data)):
                if data[j] in whites:
                    num_whites += 1
                else:
                    break
            white_after_char[index] = num_whites
            index += 1
    return white_after_char


class ReaderTestData(Dataset):
    # get the input data item for the reader model
    def __init__(self, tokenizer, samples, max_len,
                 max_num_candidates,
                 add_topic=True, use_title=False):
        self.tokenizer = tokenizer
        self.samples = samples
        self.max_len = max_len
        self.max_num_candidates = max_num_candidates
        self.add_topic = add_topic
        self.use_title = use_title
        # self.TT = '[unused1]'
        # '[unused1]' encode
        self.TT = [2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # return input_ids, attention_masks, token_type_ids, start_positions,
        # end_positions, answer_mask
        sample = self.samples[index]
        title = None
        if self.add_topic:
            title = sample['topic'] if not self.use_title else sample['title']
        encoded_pairs, attention_masks, \
        type_marks, answer_masks = self.prepare_inputs(sample, title)
        return encoded_pairs, attention_masks, type_marks, answer_masks

    def prepare_inputs(self, sample, title=None):
        mention_ids = sample['text']
        if self.add_topic:
            title_ids = self.TT + title
        else:
            title_ids = []
        candidates = sample['candidates'][:self.max_num_candidates]
        encoded_pairs = torch.zeros((self.max_num_candidates,
                                     self.max_len)).long()
        type_marks = torch.zeros((self.max_num_candidates, self.max_len)).long()
        attention_masks = torch.zeros((self.max_num_candidates,
                                       self.max_len)).long()
        answer_masks = torch.zeros((self.max_num_candidates,
                                    self.max_len)).long()
        for i, candidate in enumerate(candidates):
            candidate_ids = candidate['text_ids']
            candidate_masks = candidate['text_masks']
            # CLS + mention_ids + title_ids + SEP + candidate_ids
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
        return encoded_pairs, attention_masks, type_marks, answer_masks
