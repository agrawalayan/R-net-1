# -*- coding: utf-8 -*-
#/usr/bin/python2

import cPickle as pickle
import numpy as np
import json
import codecs
import unicodedata
import re
import sys
import os
import argparse
import tensorflow as tf
import random


import ujson as json
from collections import Counter
import numpy as np
import os.path
import io
from tqdm import tqdm
from params import Params

reload(sys)
sys.setdefaultencoding('utf8')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-p','--process', default = False, type = str2bool, help='Use the coreNLP tokenizer.', required=False)
parser.add_argument('-r','--reduce_glove', default = False, type = str2bool, help='Reduce glove size.', required=False)
args = parser.parse_args()

import spacy

nlp = spacy.blank("en")


def tokenize_corenlp(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def find_answer_index(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def write_file(indices, dir_, separate = "\n"):
    with codecs.open(dir_,"ab","utf-8") as f:
        f.write(" ".join(indices) + separate)

def pad_data(data, max_word):
    padded_data = np.zeros((len(data),max_word),dtype = np.int32)
    for i,line in enumerate(data):
        for j,word in enumerate(line):
            if j >= max_word:
                break
            padded_data[i,j] = word
    return padded_data

def pad_char_len(data, max_word, max_char):
    padded_data = np.zeros((len(data), max_word), dtype=np.int32)
    for i, line in enumerate(data):
        for j, word in enumerate(line):
            if j >= max_word:
                break
            padded_data[i, j] = word if word <= max_char else max_char
    return padded_data

def pad_char_data(data, max_char, max_words):
    padded_data = np.zeros((len(data),max_words,max_char),dtype = np.int32)
    for i,line in enumerate(data):
        for j,word in enumerate(line):
            if j >= max_words:
                break
            for k,char in enumerate(word):
                if k >= max_char:
                    # ignore the rest of the word if it's longer than the limit
                    break
                padded_data[i,j,k] = char
    return padded_data

def get_char_line(line):
    line = line.split("_SPC")
    c_len = []
    chars = []
    for word in line:
        c = [int(w) for w in word.split()]
        c_len.append(len(c))
        chars.append(c)
    return chars, c_len

def load_target(dir):
    data = []
    count = 0
    with codecs.open(dir,"rb","utf-8") as f:
        line = f.readline()
        while count < 1000 if Params.mode == "debug" else line:
            line = [int(w) for w in line.split()]
            data.append(line)
            count += 1
            line = f.readline()
    return data

def load_word(dir):
    data = []
    w_len = []
    count = 0
    with codecs.open(dir,"rb","utf-8") as f:
        line = f.readline()
        while count < 1000 if Params.mode == "debug" else line:
            line = [int(w) for w in line.split()]
            data.append(line)
            count += 1
            w_len.append(len(line))
            line = f.readline()
    return data, w_len

def load_char(dir):
    data = []
    w_len = []
    c_len_ = []
    count = 0
    with codecs.open(dir,"rb","utf-8") as f:
        line = f.readline()
        while count < 1000 if Params.mode == "debug" else line:
            c_len = []
            chars = []
            line = line.split("_SPC")
            for word in line:
                c = [int(w) for w in word.split()]
                c_len.append(len(c))
                chars.append(c)
            data.append(chars)
            line = f.readline()
            count += 1
            c_len_.append(c_len)
            w_len.append(len(c_len))
    return data, c_len_, w_len

def max_value(inputlist):
    max_val = 0
    for list_ in inputlist:
        for val in list_:
            if val > max_val:
                max_val = val
    return max_val

def process_json(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = tokenize_corenlp(context)
                context_chars = [list(token) for token in context_tokens]
                spans = find_answer_index(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = tokenize_corenlp(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def process_glove(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with io.open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def realtime_process(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0

        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def main():
    if args.reduce_glove:
        word_counter, char_counter = Counter(), Counter()
        train_examples, train_eval = process_json(
            Params.data_dir + "train-v1.1.json", "train", word_counter, char_counter)
        dev_examples, dev_eval = process_json(
            Params.data_dir + "dev-v1.1.json", "dev", word_counter, char_counter)
        test_examples, test_eval = process_json(
            Params.data_dir + "dev-v1.1.json", "test", word_counter, char_counter)

        word_emb_file = Params.fasttext_file + "wiki-news-300d-1M.vec"
        char_emb_file = Params.glove_char if Params.pretrained_char else None
        char_emb_size = Params.char_vocab_size if Params.pretrained_char else None
        char_emb_dim = Params.emb_size if Params.pretrained_char else Params.char_emb_size

        word2idx_dict = None
        if os.path.isfile(Params.word2idx_file):
            with open(Params.word2idx_file, "r") as fh:
                word2idx_dict = json.load(fh)
        word_emb_mat, word2idx_dict = process_glove(word_counter, "word", emb_file=word_emb_file,
                                                    size=Params.glove_word_size, vec_size=Params.emb_size,
                                                    token2idx_dict=word2idx_dict)

        char2idx_dict = None
        if os.path.isfile(Params.char2idx_file):
            with open(Params.char2idx_file, "r") as fh:
                char2idx_dict = json.load(fh)
        char_emb_mat, char2idx_dict = process_glove(
            char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim,
            token2idx_dict=char2idx_dict)

        realtime_process(Params, train_examples, "train",
                       Params.train_record_file, word2idx_dict, char2idx_dict)
        dev_meta = realtime_process(Params, dev_examples, "dev",
                                  Params.dev_record_file, word2idx_dict, char2idx_dict)
        test_meta = realtime_process(Params, test_examples, "test",
                                   Params.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

        save(Params.word_emb_file, word_emb_mat, message="word embedding")
        save(Params.char_emb_file, char_emb_mat, message="char embedding")
        save(Params.train_eval_file, train_eval, message="train eval")
        save(Params.dev_eval_file, dev_eval, message="dev eval")
        save(Params.test_eval_file, test_eval, message="test eval")
        save(Params.dev_meta, dev_meta, message="dev meta")
        save(Params.word2idx_file, word2idx_dict, message="word2idx")
        save(Params.char2idx_file, char2idx_dict, message="char2idx")
        save(Params.test_meta, test_meta, message="test meta")

if __name__ == "__main__":
    main()
