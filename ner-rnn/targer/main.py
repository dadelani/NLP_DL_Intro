from __future__ import print_function
from math import ceil, floor
from os.path import isfile
import time
import numpy as np
import torch.nn as nn
from src.classes.report import Report
from src.classes.utils import *
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_datasets_bank import DatasetsBankFactory
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_optimizer import OptimizerFactory
from src.factories.factory_tagger import TaggerFactory
from src.seq_indexers.seq_indexer_tag import SeqIndexerTag
from src.seq_indexers.seq_indexer_word import SeqIndexerWord


def main_funct(args):
    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)
    # Load text data as lists of lists of words (sequences) and corresponding list of lists of tags
    data_io = DataIOFactory.create(args)
    word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, tag_sequences_test = data_io.read_train_dev_test(args)
    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches
    datasets_bank = DatasetsBankFactory.create(args)
    datasets_bank.add_train_sequences(word_sequences_train, tag_sequences_train)
    datasets_bank.add_dev_sequences(word_sequences_dev, tag_sequences_dev)
    datasets_bank.add_test_sequences(word_sequences_test, tag_sequences_test)
    # Word_seq_indexer converts lists of lists of words to lists of lists of integer indices and back
    if args.word_seq_indexer is not None and isfile(args.word_seq_indexer):
        word_seq_indexer = torch.load(args.word_seq_indexer)
    else:
        word_seq_indexer = SeqIndexerWord(gpu=args.gpu, check_for_lowercase=args.check_for_lowercase,
                                          embeddings_dim=args.emb_dim, verbose=True)
        word_seq_indexer.load_items_from_embeddings_file_and_unique_words_list(emb_fn=args.emb_fn,
                                                                               emb_delimiter=args.emb_delimiter,
                                                                               emb_load_all=args.emb_load_all,
                                                                               unique_words_list=datasets_bank.unique_words_list)
    if args.word_seq_indexer is not None and not isfile(args.word_seq_indexer):
        torch.save(word_seq_indexer, args.word_seq_indexer)
    # Tag_seq_indexer converts lists of lists of tags to lists of lists of integer indices and back
    tag_seq_indexer = SeqIndexerTag(gpu=args.gpu)
    tag_seq_indexer.load_items_from_tag_sequences(tag_sequences_train)
    # Create or load pre-trained tagger
    if args.load is None:
        tagger = TaggerFactory.create(args, word_seq_indexer, tag_seq_indexer, tag_sequences_train)
    else:
        tagger = TaggerFactory.load(args.load, args.gpu)
        
    # count number of parameters
    num_parameters = np.sum([p.numel() for p in tagger.parameters()])
    print(f"The model has {num_parameters} parameters.")
    
    # Create evaluator
    evaluator = EvaluatorFactory.create(args)
    # Create optimizer
    optimizer, scheduler = OptimizerFactory.create(args, tagger)
    # Prepare report and temporary variables for "save best" strategy
    report = Report(args.report_fn, args, score_names=('train loss', '%s-train' % args.evaluator,
                                                       '%s-dev' % args.evaluator, '%s-test' % args.evaluator))
    # Initialize training variables
    iterations_num = floor(datasets_bank.train_data_num / args.batch_size)
    best_dev_score = -1
    best_epoch = -1
    best_test_score = -1
    best_test_msg = 'N\A'
    patience_counter = 0
    print('\nStart training...\n')
    for epoch in range(0, args.epoch_num + 1):
        time_start = time.time()
        loss_sum = 0
        if epoch > 0:
            tagger.train()
            if args.lr_decay > 0:
                scheduler.step()
            for i, (word_sequences_train_batch, tag_sequences_train_batch) in \
                    enumerate(datasets_bank.get_train_batches(args.batch_size)):
                tagger.train()
                tagger.zero_grad()
                loss = tagger.get_loss(word_sequences_train_batch, tag_sequences_train_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(tagger.parameters(), args.clip_grad)
                optimizer.step()
                loss_sum += loss.item()
                if i % 1 == 0:
                    print('\r-- train epoch %d/%d, batch %d/%d (%1.2f%%), loss = %1.2f.' % (epoch, args.epoch_num,
                                                                                         i + 1, iterations_num,
                                                                                         ceil(i*100.0/iterations_num),
                                                                                         loss_sum*100 / iterations_num),
                                                                                         end='', flush=True)
        # Evaluate tagger
        train_score, dev_score, test_score, test_msg = evaluator.get_evaluation_score_train_dev_test(tagger,
                                                                                                     datasets_bank,
                                                                                                     batch_size=100)
        print('\n== eval epoch %d/%d "%s" train / dev / test | %1.2f / %1.2f / %1.2f.' % (epoch, args.epoch_num,
                                                                                        args.evaluator, train_score,
                                                                                        dev_score, test_score))
        report.write_epoch_scores(epoch, (loss_sum*100 / iterations_num, train_score, dev_score, test_score))
        # Save curr tagger if required
        # tagger.save('tagger_NER_epoch_%03d.hdf5' % epoch)
        # Early stopping
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_test_score = test_score
            best_epoch = epoch
            best_test_msg = test_msg
            patience_counter = 0
            if args.save is not None and args.save_best:
                tagger.save_tagger(args.save)
            print('## [BEST epoch], %d seconds.\n' % (time.time() - time_start))
        else:
            patience_counter += 1
            print('## [no improvement micro-f1 on DEV during the last %d epochs (best_f1_dev=%1.2f), %d seconds].\n' %
                                                                                            (patience_counter,
                                                                                             best_dev_score,
                                                                                             (time.time()-time_start)))
        if patience_counter > args.patience and epoch > args.min_epoch_num:
            break
    # Save final trained tagger to disk, if it is not already saved according to "save best"
    if args.save is not None and not args.save_best:
        tagger.save_tagger(args.save)
    # Show and save the final scores
    if args.save_best:
        report.write_final_score('Final eval on test, "save best", best epoch on dev %d, %s, test = %1.2f)' %
                                 (best_epoch, args.evaluator, best_test_score))
        report.write_msg(best_test_msg)
        print("Here1")
        report.write_input_arguments()
        print("Here2")
        report.write_final_line_score(best_test_score)
    else:
        report.write_final_score('Final eval on test, %s test = %1.2f)' % (args.evaluator, test_score))
        report.write_msg(test_msg)
        report.write_input_arguments()
        report.write_final_line_score(test_score)
    if args.verbose:
        report.make_print()
