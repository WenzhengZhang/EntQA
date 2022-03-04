import torch
from retriever import DualEncoder
import argparse
import numpy as np
import os
import random
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, \
    get_linear_schedule_with_warmup, get_constant_schedule
from datetime import datetime
import json
from collections import OrderedDict
from utils import Logger, strtime
from sklearn.metrics import label_ranking_average_precision_score
from data_retriever import load_data, get_loaders, \
    get_embeddings, get_hard_negative, save_candidates, get_labels, \
    get_entity_map, get_loader_from_candidates


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def load_model(is_init, config_path, model_path, device, type_loss,
               blink=True):
    with open(config_path) as json_file:
        params = json.load(json_file)
    if blink:
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params["bert_model"])
    else:
        ctxt_bert = BertModel.from_pretrained('bert-large-uncased')
        cand_bert = BertModel.from_pretrained('bert-large-uncased')
    state_dict = torch.load(model_path) if device.type == 'cuda' else \
        torch.load(model_path, map_location=torch.device('cpu'))
    if is_init:
        if blink:
            ctxt_dict = OrderedDict()
            cand_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:26] == 'context_encoder.bert_model':
                    new_k = k[27:]
                    ctxt_dict[new_k] = v
                if k[:23] == 'cand_encoder.bert_model':
                    new_k = k[24:]
                    cand_dict[new_k] = v
            ctxt_bert.load_state_dict(ctxt_dict, strict=False)
            cand_bert.load_state_dict(cand_dict, strict=False)
        model = DualEncoder(ctxt_bert, cand_bert, type_loss)
    else:
        model = DualEncoder(ctxt_bert, cand_bert, type_loss)
        model.load_state_dict(state_dict['sd'])
    return model


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      eps=args.adam_epsilon)

    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_simple(args, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = 0

    scheduler = get_constant_schedule(optimizer)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def evaluate(scores_k, top_k,
             labels):
    # return modified hard recall@k, lrap and recall@K
    # hard recall: predict successfully if all labels are predicted
    #  recall: micro over passages
    nb_samples = len(labels)
    r_k = 0
    y_trues = []
    num_ents = 0
    num_hits = 0
    preds = []
    assert len(labels) == top_k.shape[0]
    for i in range(len(labels)):
        label = labels[i]
        pred = top_k[i]
        preds.append(pred)
        r_k += set(label).issubset(set(pred))
        y_trues.append(np.in1d(pred, label))
        num_ents += len(set(label))
        num_hits += len(set(label).intersection(set(pred)))
    r_k /= nb_samples
    h_k = num_hits / num_ents
    y_trues = np.vstack(y_trues)
    lrap = label_ranking_average_precision_score(y_trues, scores_k)
    return r_k, lrap, h_k


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    start_time = datetime.now()
    set_seeds(args)
    # configure logger
    best_val_perf = float('-inf')
    logger = Logger(args.model + '.log', on=True)
    logger.log(str(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    logger.log(f'Using device: {str(device)}', force=True)

    # load data and initialize model and dataset
    samples_train, samples_val, samples_test, entities = \
        load_data(args.data_dir, args.kb_dir)
    logger.log('number of entities {:d}'.format(len(entities)))
    # get model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    max_num_positives = args.k - args.num_cands
    config = {
        "top_k": 100,
        "biencoder_model": args.pretrained_path + "biencoder_wiki_large.bin",
        "biencoder_config": args.pretrained_path + "biencoder_wiki_large.json"
    }
    model = load_model(True, config['biencoder_config'],
                       config['biencoder_model'], device, args.type_loss,
                       args.blink)
    # configure optimizer
    num_train_samples = len(samples_train)
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, num_train_samples)
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, num_train_samples)
    if args.resume_training:
        cpt = torch.load(args.model) if device.type == 'cuda' \
            else torch.load(args.model, map_location=torch.device('cpu'))
        model.load_state_dict(cpt['sd'])
        optimizer.load_state_dict(cpt['opt_sd'])
        scheduler.load_state_dict(cpt['scheduler_sd'])
        best_val_perf = cpt['perf']
    model.to(device)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level)
    args.n_gpu = torch.cuda.device_count()
    dp = args.n_gpu > 1
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)
    train_men_loader, val_men_loader, test_men_loader, entity_loader = \
        get_loaders(samples_train, samples_val, samples_test, entities,
                    args.max_len, tokenizer, args.mention_bsz,
                    args.entity_bsz, args.add_topic, args.use_title)
    entity_map = get_entity_map(entities)
    train_labels = get_labels(samples_train, entity_map)
    val_labels = get_labels(samples_val, entity_map)
    test_labels = get_labels(samples_test, entity_map)
    model.train()
    effective_bsz = args.B * args.gradient_accumulation_steps
    # train
    logger.log('***** train *****')
    logger.log('# train samples: {:d}'.format(num_train_samples))
    logger.log('# val samples: {:d}'.format(len(samples_val)))
    logger.log('# test samples: {:d}'.format(len(samples_test)))
    logger.log('# epochs: {:d}'.format(args.epochs))
    logger.log(' batch size : {:d}'.format(args.B))
    logger.log(' gradient accumulation steps {:d}'
               ''.format(args.gradient_accumulation_steps))
    logger.log(
        ' effective training batch size with accumulation: {:d}'
        ''.format(effective_bsz))
    logger.log(' # training steps: {:d}'.format(num_train_steps))
    logger.log(' # warmup steps: {:d}'.format(num_warmup_steps))
    logger.log(' learning rate: {:g}'.format(args.lr))
    logger.log(' # parameters: {:d}'.format(count_parameters(model)))

    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    start_epoch = 1
    if args.resume_training:
        step_num = cpt['step_num']
        tr_loss, logging_loss = cpt['tr_loss'], cpt['logging_loss']
        start_epoch = cpt['epoch'] + 1
    model.zero_grad()
    all_cands_embeds = None
    logger.log('get candidates embeddings')
    if args.resume_training or args.epochs == 0:
        # we store candidates embeddings after each epoch
        all_cands_embeds = np.load(args.cands_embeds_path)
    elif args.rands_ratio != 1.0 and args.epochs != 0:
        all_cands_embeds = get_embeddings(entity_loader, model, False, device)

    for epoch in range(start_epoch, args.epochs + 1):
        logger.log('\nEpoch {:d}'.format(epoch))
        epoch_start_time = datetime.now()

        if args.rands_ratio == 1.0:
            logger.log('no need to mine hard negatives')
            candidates = None
        else:
            mention_embeds = get_embeddings(train_men_loader, model, True,
                                            device)
            logger.log('mining hard negatives')
            mining_start_time = datetime.now()
            candidates = get_hard_negative(mention_embeds, all_cands_embeds,
                                           args.num_cands,
                                           max_num_positives,
                                           args.use_gpu_index)[0]
            mining_time = strtime(mining_start_time)
            logger.log('mining time for epoch {:3d} '
                       'are {:s}'.format(epoch, mining_time))
        train_loader = get_loader_from_candidates(samples_train, entities,
                                                  train_labels, args.max_len,
                                                  tokenizer, candidates,
                                                  args.num_cands,
                                                  args.rands_ratio,
                                                  args.type_loss,
                                                  args.add_topic,
                                                  args.use_title, True, args.B)
        epoch_train_start_time = datetime.now()
        for step, batch in enumerate(train_loader):
            model.train()
            bsz = batch[0].size(0)
            batch = tuple(t.to(device) for t in batch)
            loss = model(*batch)[0]
            if dp:
                loss = loss.sum() / bsz
            else:
                loss /= bsz
            loss_avg = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss_avg, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_avg.backward()
            tr_loss += loss_avg.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                   args.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step_num += 1

                if step_num % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    logger.log('Step {:10d}/{:d} | Epoch {:3d} | '
                               'Batch {:5d}/{:5d} | '
                               'Average Loss {:8.4f}'
                               ''.format(step_num, num_train_steps,
                                         epoch, step + 1,
                                         len(train_loader), avg_loss))
                    logging_loss = tr_loss

        logger.log('training time for epoch {:3d} '
                   'is {:s}'.format(epoch, strtime(epoch_train_start_time)))
        all_cands_embeds = get_embeddings(entity_loader, model, False, device)
        all_mention_embeds = get_embeddings(val_men_loader, model, True, device)
        top_k, scores_k = get_hard_negative(all_mention_embeds,
                                            all_cands_embeds, args.k,
                                            0, args.use_gpu_index)
        eval_result = evaluate(scores_k, top_k, val_labels)

        logger.log('Done with epoch {:3d} | train loss {:8.4f} | '
                   'validation hard recall {:8.4f}'
                   '|validation LRAP {:8.4f} | validation recall {:8.4f}|'
                   ' epoch time {} '.format(
            epoch,
            tr_loss / step_num,
            eval_result[0],
            eval_result[1],
            eval_result[2],
            strtime(epoch_start_time)
        ))
        save_model = (eval_result[2] >= best_val_perf)
        if save_model:
            current_best = eval_result[2]
            logger.log('------- new best val perf: {:g} --> {:g} '
                       ''.format(best_val_perf, current_best))
            best_val_perf = current_best
            torch.save({'opt': args,
                        'sd': model.module.state_dict() if dp else model.state_dict(),
                        'perf': best_val_perf, 'epoch': epoch,
                        'opt_sd': optimizer.state_dict(),
                        'scheduler_sd': scheduler.state_dict(),
                        'tr_loss': tr_loss, 'step_num': step_num,
                        'logging_loss': logging_loss},
                       args.model)
            np.save(args.cands_embeds_path, all_cands_embeds)
        else:
            logger.log('')
    model = load_model(False, config['biencoder_config'],
                       args.model, device,
                       args.type_loss,
                       args.blink)
    model.to(device)
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)
    model.eval()
    all_cands_embeds = np.load(args.cands_embeds_path)
    logger.log('getting test mention embeddings ...')
    test_mention_embeds = get_embeddings(test_men_loader, model, True, device)
    start_time_test_infer = datetime.now()
    top_k_test, scores_k_test = get_hard_negative(test_mention_embeds,
                                                  all_cands_embeds,
                                                  args.k, 0, args.use_gpu_index)
    logger.log('test inference time {:s}'
               ''.format(strtime(start_time_test_infer)))
    test_result = evaluate(scores_k_test,
                           top_k_test, test_labels)
    logger.log(' test hard recall@{:d} : {:8.4f}'
               '| test LRAP : {:8.4f}| '
               'test recall : {:8.4f}| '
               ''.format(args.k,
                         test_result[0],
                         test_result[1],
                         test_result[2]))
    logger.log('saving test pairs')
    save_candidates(samples_test, top_k_test, entity_map, test_labels,
                    args.out_dir, 'test')
    val_mention_embeds = get_embeddings(val_men_loader, model, True, device)
    start_time_val_infer = datetime.now()
    top_k_val, scores_k_val = get_hard_negative(val_mention_embeds,
                                                all_cands_embeds, args.k, 0,
                                                args.use_gpu_index)
    logger.log('val inference time {:s} |'
               'val infer time per instance {:s}'
               ''.format(strtime(start_time_val_infer),
                         str((datetime.now() - start_time_val_infer) / len(
                             samples_val))))
    val_result = evaluate(scores_k_val,
                          top_k_val, val_labels)
    logger.log(' val hard recall@{:d} : {:8.4f}'
               '| val LRAP : {:8.4f}| '
               'val recall : {:8.4f}| '
               ''.format(args.k,
                         val_result[0],
                         val_result[1],
                         val_result[2]))
    logger.log('saving val pairs')
    save_candidates(samples_val, top_k_val, entity_map, val_labels,
                    args.out_dir, 'val')
    train_mention_embeds = get_embeddings(train_men_loader, model, True,
                                          device)
    top_k_train, scores_k_train = get_hard_negative(train_mention_embeds,
                                                    all_cands_embeds, args.k,
                                                    0, args.use_gpu_index)
    logger.log('saving train pairs')
    save_candidates(samples_train, top_k_train, entity_map,
                    train_labels,
                    args.out_dir,
                    'train')
    logger.log('experiments time {:s}'.format(strtime(start_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model path')
    parser.add_argument('--pretrained_path', type=str,
                        help='the directory of the wikipedia pretrained models')
    parser.add_argument('--resume_training', action='store_true',
                        help='resume training from checkpoint?')
    parser.add_argument('--type_loss', type=str,
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='type of multi-label loss ?')
    parser.add_argument('--use_title', action='store_true',
                        help='use title or topic?')
    parser.add_argument('--add_topic', action='store_true',
                        help='add topic information?')
    parser.add_argument('--blink', action='store_true',
                        help='use BLINK pretrained model?')
    parser.add_argument('--max_len', type=int, default=100,
                        help='max length of the mention input ')
    parser.add_argument('--data_dir', type=str,
                        help='the  data directory')
    parser.add_argument('--kb_dir', type=str,
                        help='the knowledge base directory')
    parser.add_argument('--out_dir', type=str,
                        help='the output saving directory')
    parser.add_argument('--B', type=int, default=16,
                        help='the batch size per gpu')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='the learning rate')
    parser.add_argument('--epochs', type=int, default=3,
                        help='the number of training epochs')
    parser.add_argument('--k', type=int, default=100,
                        help='recall@k when evaluate')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='proportion of training steps to perform linear '
                             'learning rate warmup for [%(default)g]')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay [%(default)g]')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer [%(default)g]')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='num gradient accumulation steps [%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers [%(default)d]')
    parser.add_argument('--simpleoptim', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='num logging steps [%(default)d]')
    parser.add_argument('--gpus', default='', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--rands_ratio', default=1.0, type=float,
                        help='the ratio of random candidates and hard')
    parser.add_argument('--num_cands', default=64, type=int,
                        help='the total number of candidates')
    parser.add_argument('--mention_bsz', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--entity_bsz', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--use_gpu_index', action='store_true',
                        help='use gpu index?')
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) "
             "instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', "
             "'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument('--cands_embeds_path', type=str,
                        help='the directory of candidates embeddings')
    parser.add_argument('--use_cached_embeds', action='store_true',
                        help='use cached candidates embeddings ?')
    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior
    main(args)
