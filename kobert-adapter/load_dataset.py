import csv
import torch
import numpy as np
import gluonnlp as nlp

from KoBERT.kobert.utils import get_tokenizer

max_len = 64
pad = True
pair = False

def load_nsmc_train(vocab):
    train_dir = './Dataset/nsmc/ratings_train.txt'

    train = []
    
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len, pad=pad, pair=pair)

    with open(train_dir, 'r') as tr:
        reader = csv.reader(tr, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            newrow = []
            sentence = row[1]
            label = row[2]
            newrow.append(transform([sentence]))
            newrow.append(np.int32(label))
            train.append(newrow)

    return train

def load_nsmc_test(vocab):
    test_dir = './Dataset/nsmc/ratings_test.txt'

    test = []
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len, pad=pad, pair=pair)

    with open(test_dir, 'r') as te:
        reader = csv.reader(te, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            newrow = []
            try:
                sentence = row[1]
                label = row[2]
                newrow.append(transform([sentence]))
                newrow.append(np.int32(label))
                test.append(newrow)
            except:
                continue

    return test

def load_nsmc_train_part(vocab):
    train_dir = './Dataset/nsmc/ratings_train.txt'

    train = []
    
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len, pad=pad, pair=pair)

    with open(train_dir, 'r') as tr:
        reader = csv.reader(tr, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            newrow = []
            sentence = row[1]
            label = row[2]
            newrow.append(transform([sentence]))
            newrow.append(np.int32(label))
            train.append(newrow)
            if idx == 1500:
                break

    return train

def load_4way_train(vocab):
    train_dir = './Dataset/4way/final_remove_dup_no_neutral_train.csv'

    train = []
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len, pad=pad, pair=pair)

    with open(train_dir, 'r') as te:
        reader = csv.reader(te, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            newrow = []
            sentence = row[2]
            label = int(row[1]) -1

            newrow.append(transform([sentence]))
            newrow.append(np.int32(label))
            train.append(newrow)


    return train

def load_4way_test(vocab):
    test_dir = './Dataset/4way/final_remove_dup_no_neutral_test.csv'

    test = []
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len, pad=pad, pair=pair)

    with open(test_dir, 'r') as te:
        reader = csv.reader(te, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            newrow = []
            sentence = row[2]
            label = int(row[1]) -1

            newrow.append(transform([sentence]))
            newrow.append(np.int32(label))
            test.append(newrow)

    return test
