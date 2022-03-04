import argparse
import os
import torch
import torch.nn as nn
from data_reader import load_data, get_loaders, get_golds, \
    get_results_doc, save_results
from datetime import datetime
from utils import Logger, compute_strong_micro_results, strtime
from reader import Reader, get_predicts, prune_predicts
from transformers import BertTokenizer, BertModel, ElectraModel, \
    ElectraTokenizer
from sklearn.metrics import label_ranking_average_precision_score
from run_retriever import configure_optimizer, configure_optimizer_simple, \
    set_seeds
import pickle as pkl


def get_raw_results(model, device, loader, k, samples,
                    do_rerank,
                    filter_span=True,
                    no_multi_ents=False):
    model.eval()
    ranking_scores = []
    ranking_labels = []
    ps = []
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            if do_rerank:
                batch_p, rank_logits_b = model(*batch)
            else:
                batch_p = model(*batch).detach()
            batch_p = batch_p.cpu()
            ps.append(batch_p)
            if do_rerank:
                ranking_scores.append(rank_logits_b.cpu())
                ranking_labels.append(batch[4].cpu())
        ps = torch.cat(ps, 0)
    raw_predicts = get_predicts(ps, k, filter_span, no_multi_ents)
    assert len(raw_predicts) == len(samples)
    if do_rerank:
        ranking_scores = torch.cat(ranking_scores, 0)
        ranking_labels = torch.cat(ranking_labels, 0)
    else:
        ranking_scores = None
        ranking_labels = None
    return raw_predicts, ranking_scores, ranking_labels


def transform_predicts(preds, entities, samples):
    #  ent_idx,start,end --> start, end, ent name
    ent_titles = [e['title'] for e in entities]
    assert len(preds) == len(samples)
    results = []
    for ps, s in zip(preds, samples):
        results_p = []
        for p in ps:
            ent_title = ent_titles[s['candidates'][p[0]]]
            r = p[1:]
            # start, end, entity name
            r.append(ent_title)
            results_p.append(r)
        results.append(results_p)
    return results


def evaluate_rerank(rank_scores, rank_labels):
    ranking_scores = rank_scores.numpy()
    ranking_labels = rank_labels.numpy()
    rank_lrap = label_ranking_average_precision_score(ranking_labels,
                                                      ranking_scores)
    return rank_lrap


def evaluate_after_prune(logger, pruned_preds, golds,
                         samples):
    predicts_doc = get_results_doc(pruned_preds, samples)
    precision, recall, f_1 = compute_strong_micro_results(predicts_doc, golds,
                                                          logger)
    return {'precision': precision, 'recall': recall, 'F1': f_1}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(is_init, model_path, type_encoder, device, type_span_loss,
               do_rerank, type_rank_loss, max_answer_len, max_passage_len):
    if is_init:
        encoder, tokenizer = get_encoder(type_encoder, True)
        model = Reader(encoder, type_span_loss, do_rerank, type_rank_loss,
                       max_answer_len, max_passage_len)
        return model, tokenizer
    else:
        encoder = get_encoder(type_encoder, False)
        package = torch.load(model_path) if device.type == 'cuda' else \
            torch.load(model_path, map_location=torch.device('cpu'))
        model = Reader(encoder, type_span_loss, do_rerank, type_rank_loss,
                       max_answer_len, max_passage_len)
        try:
            model.load_state_dict(package['sd'])
        except RuntimeError:
            # forgot to save model.module.sate_dict
            from collections import OrderedDict
            state_dict = package['sd']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                # for loading our old version reader model
                if name != 'topic_query':
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        return model


def get_encoder(type_encoder, return_tokenizer=False):
    if type_encoder == 'bert_base':
        encoder = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif type_encoder == 'bert_large':
        encoder = BertModel.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    elif type_encoder == 'electra_base':
        encoder = ElectraModel.from_pretrained(
            'google/electra-base-discriminator')
        tokenizer = ElectraTokenizer.from_pretrained(
            'google/electra-base-discriminator')
    elif type_encoder == 'electra_large':
        encoder = ElectraModel.from_pretrained(
            'google/electra-large-discriminator')
        tokenizer = ElectraTokenizer.from_pretrained(
            'google/electra-large-discriminator')
    elif type_encoder == 'squad2_bert_large':
        encoder = BertModel.from_pretrained(
            "phiyodr/bert-large-finetuned-squad2")
        tokenizer = BertTokenizer.from_pretrained(
            "phiyodr/bert-large-finetuned-squad2")
    elif type_encoder == 'squad2_electra_large':
        encoder = ElectraModel.from_pretrained(
            'ahotrod/electra_large_discriminator_squad2_512')
        tokenizer = ElectraTokenizer.from_pretrained(
            'ahotrod/electra_large_discriminator_squad2_512')
    else:
        raise ValueError('wrong encoder type')
    if return_tokenizer:
        return encoder, tokenizer
    else:
        return encoder


def main(args):
    set_seeds(args)
    # configure logger
    best_val_perf = float('-inf')
    logger = Logger(args.model + '.log', on=True)
    logger.log(str(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {str(device)}', force=True)
    # load data and get dataloaders
    data = load_data(args.data_dir, args.kb_dir)
    train_golds_doc, val_golds_doc, test_golds_doc, p_train_golds, \
    p_val_golds, p_test_golds = get_golds(data[0], data[1], data[2])

    # get model and tokenizer
    model, tokenizer = load_model(True, args.model, args.type_encoder, device,
                                  args.type_span_loss, args.do_rerank,
                                  args.type_rank_loss, args.max_answer_len,
                                  args.max_passage_len)
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, len(data[0]))
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, len(data[0]))
    if args.resume_training:
        cpt = torch.load(args.model) if device.type == 'cuda' \
            else torch.load(args.model, map_location=torch.device('cpu'))
        model.load_state_dict(cpt['sd'])
        optimizer.load_state_dict(cpt['opt_sd'])
        scheduler.load_state_dict(cpt['scheduler_sd'])
        best_val_perf = cpt['perf']
    model.to(device)
    loader_train, loader_dev, loader_test = get_loaders(tokenizer, data, args.L,
                                                        args.C, args.C_val,
                                                        args.B, args.val_bsz,
                                                        args.add_topic,
                                                        args.use_title)

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
    effective_bsz = args.B * args.gradient_accumulation_steps
    logger.log('\n[TRAIN]')
    logger.log('  # train samples: %d' % len(data[0]))
    logger.log('  # dev samples: %d' % len(data[1]))
    logger.log('  # test samples: %d' % len(data[2]))
    logger.log('  # epochs:        %d' % args.epochs)
    logger.log('  batch size:      %d' % args.B)
    logger.log('  grad accum steps %d' % args.gradient_accumulation_steps)
    logger.log('    (effective batch size w/ accumulation: %d)' % effective_bsz)
    logger.log('  # train steps:   %d' % num_train_steps)
    logger.log('  # warmup steps:  %d' % num_warmup_steps)
    logger.log('  learning rate:   %g' % args.lr)
    logger.log('  # parameters:    %d' % count_parameters(model))

    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    start_epoch = 1
    if args.resume_training:
        step_num = cpt['step_num']
        tr_loss, logging_loss = cpt['tr_loss'], cpt['logging_loss']
        start_epoch = cpt['epoch'] + 1
    model.train()
    model.zero_grad()
    start_time = datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        logger.log('\nEpoch %d' % epoch)
        start_time_epoch = datetime.now()
        for step, batch in enumerate(loader_train):  # Shuffled every epoch
            model.train()
            bsz = batch[0].size(0)
            batch = tuple(t.to(device) for t in batch)
            loss = model(*batch)
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
                               'Average Loss {:8.4f}'.format(
                        step_num, num_train_steps, epoch,
                        step, len(loader_train), avg_loss))
                    logging_loss = tr_loss
        logger.log('training time for epoch {:3d} is '
                   '{:s}'.format(epoch, strtime(start_time_epoch)))
        logger.log('validating...')

        val_raw_predicts, val_rank_scores, val_rank_labels = get_raw_results(
            model, device, loader_dev,
            args.k, data[1], args.do_rerank,
            args.filter_span,
            args.no_multi_ents)
        pruned_val_preds = prune_predicts(val_raw_predicts, args.thresd)
        val_predicts = transform_predicts(pruned_val_preds, data[-1],
                                          data[1])
        val_result = evaluate_after_prune(logger, val_predicts,
                                          val_golds_doc, data[1])

        logger.log('Done with epoch {:3d} | train loss {:8.4f} | '
                   'val recall  {} | '
                   'val precision {} |'
                   'val F1 {} |'.format(
            epoch,
            tr_loss / step_num,
            val_result['recall'],
            val_result['precision'],
            val_result['F1'],
            newline=False))
        if args.do_rerank:
            val_lrap = evaluate_rerank(val_rank_scores, val_rank_labels)
            logger.log('val LRAP {}'.format(val_lrap))
        if val_result['F1'] > best_val_perf:
            logger.log('      <----------New best val perf: %g -> %g' %
                       (best_val_perf, val_result['F1']))
            best_val_perf = val_result['F1']
            torch.save({'opt': args,
                        'sd': model.module.state_dict() if dp else model.state_dict(),
                        'perf': best_val_perf,
                        'val_thresd': args.thresd}, args.model)
        else:
            logger.log('')
    model = load_model(False, args.model, args.type_encoder,
                       device, args.type_span_loss, args.do_rerank,
                       args.type_rank_loss, args.max_answer_len,
                       args.max_passage_len)
    model.to(device)
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)
    model.eval()
    logger.log('getting test raw predicts')
    start_time_test_infer = datetime.now()
    test_raw_predicts, test_rank_scores, test_rank_labels = get_raw_results(
        model, device, loader_test,
        args.k, data[2], args.do_rerank,
        args.filter_span,
        args.no_multi_ents)

    logger.log('prune and evaluate test...')
    pruned_test_preds = prune_predicts(test_raw_predicts, args.thresd)
    test_predicts = transform_predicts(pruned_test_preds, data[-1],
                                       data[2])
    logger.log('test inference time {:s}'.format(strtime(
        start_time_test_infer)))
    logger.log('per val instance inference time {:s}'.format(str((
            (datetime.now() - start_time_test_infer) / len(data[2])))))

    logger.log('save test results')
    test_save_path = os.path.join(args.results_dir, 'test_raw')
    with open(test_save_path, 'wb') as f:
        pkl.dump(test_raw_predicts, f)
    save_results(test_predicts, p_test_golds, data[2], args.results_dir,
                 'test')
    test_result = evaluate_after_prune(logger, test_predicts,
                                       test_golds_doc, data[2])
    logger.log('\nDone training | training time {:s} | '
               'test recall  {:8.4f}| '
               'test precision {} | '
               'test F1 {} | '.format(strtime(start_time),
                                      test_result['recall'],
                                      test_result['precision'],
                                      test_result['F1'])
               )
    if args.do_rerank:
        test_lrap = evaluate_rerank(test_rank_scores, test_rank_labels)
        logger.log('test LRAP {}'.format(test_lrap))
    logger.log('getting val raw predicts')
    start_time_val_infer = datetime.now()
    val_raw_predicts, val_rank_scores, val_rank_labels = get_raw_results(
        model, device, loader_dev,
        args.k, data[1], args.do_rerank,
        args.filter_span,
        args.no_multi_ents)
    logger.log('prune and evaluate val ...')
    pruned_val_preds = prune_predicts(val_raw_predicts, args.thresd)
    val_predicts = transform_predicts(pruned_val_preds, data[-1],
                                      data[1])
    logger.log('val inference time {:s}'.format(strtime(
        start_time_val_infer)))
    logger.log('per val instance inference time {:s}'.format(str((
            (datetime.now() - start_time_val_infer) / len(data[1])))))
    logger.log('save val results')
    val_save_path = os.path.join(args.results_dir, 'val_raw')
    with open(val_save_path, 'wb') as f:
        pkl.dump(val_raw_predicts, f)
    save_results(val_predicts, p_val_golds, data[1], args.results_dir,
                 'val')
    val_result = evaluate_after_prune(logger, val_predicts,
                                      val_golds_doc, data[1])

    logger.log('val recall  {} | '
               'val precision {} |'
               'val F1 {} |'.format(
        val_result['recall'],
        val_result['precision'],
        val_result['F1'],
        newline=False))
    if args.do_rerank:
        val_lrap = evaluate_rerank(val_rank_scores, val_rank_labels)
        logger.log('val LRAP {}'.format(val_lrap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model path')
    parser.add_argument('--data_dir', type=str,
                        help=' data directory')
    parser.add_argument('--kb_dir', type=str,
                        help=' kb directory')
    parser.add_argument('--results_dir', type=str,
                        help=' results directory')
    parser.add_argument('--L', type=int, default=160,
                        help='max length of joint input [%(default)d]')
    parser.add_argument('--max_passage_len', type=int, default=32,
                        help='max length of passage [%(default)d]')

    parser.add_argument('--filter_span', action='store_true',
                        help='filter span?')
    parser.add_argument('--resume_training', action='store_true',
                        help='resume training?')
    parser.add_argument('--no_multi_ents', action='store_true',
                        help='prevent multiple entities for a mention span?')
    parser.add_argument('--add_topic', action='store_true',
                        help='add title?')
    parser.add_argument('--do_rerank', action='store_true',
                        help='do rerank multi-tasking?')
    parser.add_argument('--stride', type=int, default=16,
                        help='passage stride [%(default)d]')
    parser.add_argument('--max_answer_len', type=int, default=10,
                        help='max length of answer [%(default)d]')
    parser.add_argument('--k', type=int, default=10,
                        help='get top-k spans per entity before top-p '
                             'filtering')
    parser.add_argument('--thresd', type=float, default=0.05,
                        help='probabilty threshold for top-p filtering')
    parser.add_argument('--num_answers', type=int, default=10,
                        help='max number of answers [%(default)d]')
    parser.add_argument('--use_title', action='store_true',
                        help='use title or use topic?')
    parser.add_argument('--random_positive', action='store_true',
                        help='random positive?')
    parser.add_argument('--oracle', action='store_true',
                        help='oracle evaluation ?')
    parser.add_argument('--C', type=int, default=64,
                        help='max number of candidates [%(default)d]')
    parser.add_argument('--C_val', type=int, default=100,
                        help='max number of candidates [%(default)d] when eval')
    parser.add_argument('--B', type=int, default=16,
                        help='batch size [%(default)d]')
    parser.add_argument('--val_bsz', type=int, default=72,
                        help='batch size [%(default)d]')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='initial learning rate [%(default)g]')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='proportion of training steps to perform linear '
                             'learning rate warmup for [%(default)g]')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay [%(default)g]')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer [%(default)g]')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='num gradient accumulation steps [%(default)d]')
    parser.add_argument('--epochs', type=int, default=3,
                        help='max number of epochs [%(default)d]')
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='num logging steps [%(default)d]')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--init', type=float, default=0,
                        help='init (default if 0) [%(default)g]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num workers [%(default)d]')
    parser.add_argument('--gpus', default='', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--simpleoptim', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--type_encoder', type=str,
                        default='bert_base',
                        help='the type of encoder')
    parser.add_argument('--type_span_loss', type=str,
                        default='sum_log',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='type of multi-label loss for span ?')
    parser.add_argument('--type_rank_loss', type=str,
                        default='sum_log',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='type of multi-label loss  for rerank?')
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

    args = parser.parse_args()

    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior

    main(args)
