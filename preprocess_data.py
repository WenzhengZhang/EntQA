# author: Wenyue Hua
import argparse
import json
import math
import random
import csv
from transformers import BertTokenizer
import os
from copy import deepcopy


def compute_length(text_list, word_length):
    length = 0
    for token in text_list[:-word_length]:
        if token != " ":
            length += len(token)
    return length


def process_raw_aida(raw_dir, part):
    full_data = []
    raw_path = os.path.join(raw_dir, "aida-yago2-dataset-{}.tsv".format(part))
    with open(raw_path, "r") as f:
        delimiter = ',' if part == 'train' else '\t'
        csvreader = csv.reader(f, delimiter=delimiter)
        # '\t' for val/test, ',' for train
        quoteCharSeenBefore = False
        # whiteSpaceInFront = True
        whiteSpaceBehind = True
        new_doc = {"doc_id": None, "text": "", "spans": [], "entities": []}
        for data in csvreader:
            if len(data) > 0:
                if data[0].startswith("-DOCSTART-"):
                    rest = data[0].replace("-DOCSTART- (", "")
                    doc_id = (
                        rest[: rest.index(" ")]
                            .replace("testa", "")
                            .replace("testb", "")
                    )
                    if new_doc["text"]:
                        full_data.append(new_doc)
                    new_doc = {
                        "doc_id": doc_id,
                        "text": "",
                        "spans": [],
                        "entities": [],
                    }
                    quoteCharSeenBefore = False
                else:
                    if data[0] != "":
                        char = data[0].replace("\n", " ").strip()
                        # char = data[0].strip()
                        # if we should insert a white space
                        whiteSpaceInFront = whiteSpaceBehind
                        whiteSpaceBehind = True
                        if len(new_doc["text"]) > 0 and len(char) >= 1:
                            if len(char) == 1:
                                if char in ["?", "!", ",", ".", ")", "]", "}"]:
                                    whiteSpaceInFront = False
                                elif char == '"':
                                    if quoteCharSeenBefore:
                                        whiteSpaceInFront = False
                                    if not quoteCharSeenBefore:
                                        whiteSpaceBehind = False
                                    quoteCharSeenBefore = not quoteCharSeenBefore
                                elif char in ["(", "[", "{"]:
                                    whiteSpaceBehind = False
                            else:
                                if not (char[0].isalpha() or char[0].isdigit()):
                                    whiteSpaceInFront = False
                                else:
                                    whiteSpaceInFront = True
                            if whiteSpaceInFront:
                                new_doc["text"] += " "
                        new_doc["text"] += char

                        if len(data) > 1:
                            if data[1] == "B" and data[3] != "--NME--":
                                word_length = len(data[0])
                                current_text_length = compute_length(
                                    new_doc["text"], word_length
                                )
                                new_doc["spans"].append(
                                    (current_text_length,
                                     len(data[2].replace(" ", "")))
                                )
                                new_doc["entities"].append(data[3])
        if new_doc["text"]:
            full_data.append(new_doc)
    return full_data


def load_processed_aida(args, part):
    path = os.path.join(args.out_processed_dir, 'aida_%s.json' % part)
    with open(path) as f:
        res = json.load(f)
    return res


def normalize_string(s):
    s = s.replace('_', ' ')
    return eval(repr(s).replace('\\\\', '\\'))


def process_entities(processed_raw_data, args):
    title_map_path = os.path.join(args.title_map_dir, 'title_map.json')
    with open(title_map_path) as f:
        title_map = json.load(f)
    res = []
    for d in processed_raw_data:
        r = deepcopy(d)
        ents = [title_map[normalize_string(e)] if normalize_string(e) in
                                                  title_map else normalize_string(
            e) for e in d['entities']]
        r['entities'] = ents
        res.append(r)
    return res


def tokenize_original_text(processed_raw_data, tokenizer, part, args):
    data = []
    for d in processed_raw_data:
        orig_text = d["text"]
        topic = orig_text.split(' ', 1)[0].replace(',', '').replace("'s", '')
        title = orig_text.split('.', 1)[0]
        topic = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(topic))
        title = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
        orig_spans = d["spans"]
        text = tokenizer.tokenize(orig_text)
        # +1 because of [BOS]
        doc_id = d["doc_id"]
        spans = [
            (char2token(text, span[0]),
             char2token(text, span[0] + span[1] - 1) + 1,)
            for span in orig_spans
        ]
        entities = d["entities"]
        text_ids = tokenizer.convert_tokens_to_ids(text)

        content_length = args.instance_length - 2

        if len(text_ids) < content_length:
            text_ids = [101] + tokenizer.convert_tokens_to_ids(text) + [102]
            spans = [(s[0] + 1, s[1] + 1) for s in spans]
            data.append(
                {
                    "doc_id": doc_id,
                    "topic": topic,
                    "title": title,
                    "text": text_ids,
                    "spans": spans,
                    "entities": entities,
                    "offset": 0,
                }
            )
        else:
            # -2 for [BOS] and [EOS]
            for ins_num in range(math.ceil(len(text_ids) / args.stride)):
                begin = ins_num * args.stride
                end = ins_num * args.stride + content_length
                instance_ids = [101] + text_ids[begin:end] + [102]
                span_ids = [
                    spans.index(s) for s in spans if
                    begin <= s[0] and s[1] <= end
                ]
                # +1 for [BOS]
                instance_spans = [
                    (
                        spans[x][0] + 1 - begin,
                        spans[x][1] + 1 - begin,
                    )
                    for x in span_ids
                ]
                instance_entities = [entities[x] for x in span_ids]

                data.append(
                    {
                        "doc_id": doc_id,
                        "topic": topic,
                        "title": title,
                        "text": instance_ids,
                        "spans": instance_spans,
                        "entities": instance_entities,
                        "offset": begin,
                    }
                )
    if part == 'train':
        data = negative_sampling(args.pos_prop, data)

        pos = 0
        neg = 0
        for d in data:
            spans = d["spans"]
            if spans:
                pos += 1
            else:
                neg += 1

        assert pos / (pos + neg) >= args.pos_prop

    return data


# only for train
def negative_sampling(pos_prop, data):
    random.seed(10)
    pos = 0
    neg = 0
    for d in data:
        spans = d["spans"]
        if spans:
            pos += 1
        else:
            neg += 1

    sampled_data = []
    if pos / (pos + neg) < pos_prop:
        neg_need_number = (pos / pos_prop) - pos
        neg_sample_rate = neg_need_number / neg

        for d in data:
            spans = d["spans"]
            if spans:
                sampled_data.append(d)
            else:
                # discard
                if random.random() > neg_sample_rate:
                    pass
                # retain
                else:
                    sampled_data.append(d)
    else:
        sampled_data = data

    return sampled_data


def char2token(text, index):
    char2token_list = []
    for i, tok in enumerate(text):
        char2token_list += [i] * len(tok.replace("##", ""))
    return char2token_list[index]


def get_entity_window(item, tokenizer, max_ent_len):
    title = item['wikipedia_title']
    text = item['text'][1:] if len(item['text']) > 1 else item['text']
    text = ' '.join(text)
    max_ent_len -= 2  # CLS, SEP
    ENT = '[unused2]'
    title_tokens = tokenizer.tokenize(title)
    text_tokens = tokenizer.tokenize(text)
    window = (title_tokens + [ENT] + text_tokens)[:max_ent_len]
    return window


# process kilt knowledge base
def process_kilt_kb(args):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    fout = open(args.out_kb_path, 'w')
    with open(args.raw_kb_path, 'r') as f:
        for line in f:
            field = {}
            item = json.loads(line)
            window = get_entity_window(item, tokenizer, args.max_ent_len)
            entity_dict = tokenizer.encode_plus(window,
                                                add_special_tokens=True,
                                                max_ent_length=args.max_ent_len,
                                                pad_to_max_ent_length=True,
                                                truncation=True)
            field['wikipedia_id'] = item['wikipedia_id']
            field['title'] = item['wikipedia_title']
            field['text_ids'] = entity_dict['input_ids']
            field['text_masks'] = entity_dict['attention_mask']
            fout.write('%s\n' % json.dumps(field))

    fout.close()


def save_aida(data, args, part):
    out_path = os.path.join(args.out_aida_dir, 'tokenized_aida_%s.json' % part)
    with open(out_path, 'w') as f:
        json.dump(data, f)


def save_aida_processed(data, args, part):
    out_path = os.path.join(args.out_processed_dir, 'aida_%s.json' % part)
    with open(out_path, 'w') as f:
        json.dump(data, f)


def main(args):
    # process raw aida
    aida_train = process_raw_aida(args.raw_dir, 'train')
    aida_val = process_raw_aida(args.raw_dir, 'val')
    aida_test = process_raw_aida(args.raw_dir, 'test')
    aida_train = process_entities(aida_train, args)
    aida_val = process_entities(aida_val, args)
    aida_test = process_entities(aida_test, args)
    if args.save_processed:
        save_aida_processed(aida_train, args, 'train')
        save_aida_processed(aida_val, args, 'val')
        save_aida_processed(aida_test, args, 'test')
    # aida_train = load_processed_aida(args, 'train')
    # aida_val = load_processed_aida(args, 'val')
    # aida_test = load_processed_aida(args, 'test')
    # tokenize aida
    print('tokenize aida...')
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    tokenized_train = tokenize_original_text(aida_train, tokenizer, 'train',
                                             args)
    print(len(tokenized_train))
    tokenized_val = tokenize_original_text(aida_val, tokenizer, 'val', args)
    print(len(tokenized_val))
    tokenized_test = tokenize_original_text(aida_test, tokenizer, 'test', args)
    print(len(tokenized_test))
    # save aida data
    print('save tokenized aida ...')
    save_aida(tokenized_train, args, 'train')
    save_aida(tokenized_val, args, 'val')
    save_aida(tokenized_test, args, 'test')
    print('process kilt kb ...')
    process_kilt_kb(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str,
                        help='raw aida data directory')
    parser.add_argument('--title_map_dir', type=str,
                        help='title map  directory')
    parser.add_argument('--save_processed', action='store_true',
                        help='save processed raw aida data?')
    parser.add_argument('--out_aida_dir', type=str,
                        help='output aida data directory')
    parser.add_argument('--out_processed_dir', type=str,
                        help='output processed raw aida data directory')
    parser.add_argument('--raw_kb_path', type=str,
                        help='raw kilt kb path')
    parser.add_argument('--out_kb_path', type=str,
                        help='output kb path')
    parser.add_argument('--max_ent_len', type=int,
                        default=128,
                        help='maximum length of entity input')
    # twice the number of passages
    parser.add_argument(
        "--instance_length", type=int, default=32,
        help="the length of each instance"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="length of stride when chunking instances",
    )
    # in val: 1296 pos, 113 neg
    # in test: 1177 pos, 134 neg
    # in train: 5062 pos, 660 neg
    parser.add_argument(
        "--pos_prop",
        type=float,
        default=1,
        help="number of passages with entities v.s. total number of passages",
    )
    args = parser.parse_args()

    main(args)
