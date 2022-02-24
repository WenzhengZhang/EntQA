import collections
import random
import bisect
import sys
import unicodedata
import re
import json
from datetime import datetime
import os


class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


def sample_range_excluding(n, k, excluding):
    skips = [j - i for i, j in enumerate(sorted(set(excluding)))]
    s = random.sample(range(n - len(skips)), k)
    return [i + bisect.bisect_right(skips, i) for i in s]


def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(_normalize(s)))))


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def strong_matching(predictions, gold_entities):
    # list of tuples, cannot be list of lists since list is not hashable
    predicts = set(predictions)
    golds = set(gold_entities)
    # number = 0
    # for p in predictions:
    #     if p in gold_entities:
    #         number += 1

    return len(golds.intersection(predicts))


def weak_matching(predictions, gold_entities):
    number = 0
    for p in predictions:
        for g in gold_entities:
            if (
                    (set(range(p[1], p[1] + p[2])) & set(
                        range(g[1], g[1] + g[2])))
                    and p[3] == g[3]
                    and p[0] == g[0]
            ):
                number += 1

    return number


def safe_divide(a, b):
    if b != 0:
        return a / b
    else:
        return 0


def compute_strong_micro_results(predictions, gold_entities, logger=None):
    prec_total = len(set(predictions))

    gold_total = len(set(gold_entities))
    strong_correct_num = strong_matching(predictions, gold_entities)
    if logger:
        logger.log(
            json.dumps({'pred_total': prec_total, 'gold_total': gold_total,
                        'strong_correct_num': strong_correct_num}))

    strong_precision = safe_divide(strong_correct_num, prec_total)

    strong_recall = safe_divide(strong_correct_num, gold_total)

    strong_f1 = 2 * safe_divide(
        (strong_precision * strong_recall), (strong_precision + strong_recall)
    )
    return [
        round(strong_precision, 4),
        round(strong_recall, 4),
        round(strong_f1, 4),
    ]


def compute_weak_micro_results(predictions, gold_entities):
    prec_total = len(predictions)

    gold_total = len(gold_entities)

    weak_correct_num = weak_matching(predictions, gold_entities)

    weak_precision = safe_divide(weak_correct_num, prec_total)

    weak_recall = safe_divide(weak_correct_num, gold_total)

    weak_f1 = 2 * safe_divide(
        (weak_precision * weak_recall), (weak_precision + weak_recall)
    )

    return [
        round(weak_precision, 4),
        round(weak_recall, 4),
        round(weak_f1, 4),
    ]


def compute_strong_macro_results(predictions, gold_entities):
    doc_predictions = {}
    doc_gold_entities = {}
    for p in predictions:
        if p[0] not in doc_predictions.keys():
            doc_predictions[p[0]] = [p]
        else:
            doc_predictions[p[0]].append(p)
    for g in gold_entities:
        if g[0] not in doc_gold_entities.keys():
            doc_gold_entities[g[0]] = [g]
        else:
            doc_gold_entities[g[0]].append(g)

    pred_ids = set(doc_predictions.keys())
    gold_ids = set(doc_gold_entities.keys())

    strong_precisions = []
    strong_recalls = []
    strong_f1s = []
    for id in pred_ids & gold_ids:
        (strong_precision, strong_recall,
         strong_f1,) = compute_strong_micro_results(
            doc_predictions[id], doc_gold_entities[id]
        )
        strong_precisions.append(strong_precision)
        strong_recalls.append(strong_recall)
        strong_f1s.append(strong_f1)
    for id in (pred_ids - gold_ids).union(gold_ids - pred_ids):
        strong_precisions.append(0)
        strong_recalls.append(0)
        strong_f1s.append(0)

    return [
        round(safe_divide(sum(x), len(x)), 4)
        for x in [strong_precisions, strong_recalls, strong_f1s]
    ]


def compute_weak_macro_results(predictions, gold_entities):
    doc_predictions = {}
    doc_gold_entities = {}
    for p in predictions:
        if p[0] not in doc_predictions.keys():
            doc_predictions[p[0]] = [p]
        else:
            doc_predictions[p[0]].append(p)
    for g in gold_entities:
        if g[0] not in doc_gold_entities.keys():
            doc_gold_entities[g[0]] = [g]
        else:
            doc_gold_entities[g[0]].append(g)

    pred_ids = set(doc_predictions.keys())
    gold_ids = set(doc_gold_entities.keys())

    weak_precisions = []
    weak_recalls = []
    weak_f1s = []
    for id in pred_ids & gold_ids:
        (weak_precision, weak_recall, weak_f1,) = compute_weak_micro_results(
            doc_predictions[id], doc_gold_entities[id]
        )
        weak_precisions.append(weak_precision)
        weak_recalls.append(weak_recall)
        weak_f1s.append(weak_f1)
    for id in (pred_ids - gold_ids).union(gold_ids - pred_ids):
        weak_precisions.append(0)
        weak_recalls.append(0)
        weak_f1s.append(0)

    return [
        round(safe_divide(sum(x), len(x)), 4)
        for x in [weak_precisions, weak_recalls, weak_f1s, ]
    ]


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds


class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        if self.on or force:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()
