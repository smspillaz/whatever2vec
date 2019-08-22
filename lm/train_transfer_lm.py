#!/usr/bin/env python 

import argparse
import os
import json
import pickle
import logging
import random
import sys

import numpy as np

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import (
    BertForPreTraining,
    BertConfig,
    BertTokenizer,
    OpenAIGPTModel,
    OpenAIGPTTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    TransfoXLModel,
    TransfoXLTokenizer,
    XLNetConfig,
    XLNetLMHeadModel,
    XLNetTokenizer,
    XLMModel,
    XLMTokenizer
)
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

logger = logging.Logger("BERT")

MODELS = [(BertForPreTraining, BertTokenizer,   'bert-base-uncased'),
          (OpenAIGPTModel,   OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model,        GPT2Tokenizer,      'gpt2'),
          (TransfoXLModel,   TransfoXLTokenizer, 'transfo-xl-wt103'),
          (XLNetLMHeadModel, XLNetTokenizer,     'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024')]


def train_loop(model,
               optimizer,
               scheduler,
               train_loader,
               val_loader,
               epochs,
               device,
               model_type,
               max_grad_norm=1.0,
               epoch_end_callback=None,
               iteration_end_callback=None):
    epoch_end_callback = epoch_end_callback or (lambda m, s: None)
    iteration_end_callback = iteration_end_callback or (lambda m, s: None)

    for epoch in tqdm(range(epochs), desc="epoch"):
        model.train()

        train_bar = tqdm(train_loader, desc="train")
        for i, batch in enumerate(train_bar):
            batch = all_to_device(batch, device)
            # input_ids, input_mask, segment_ids, lm_label_ids, is_next = tuple(t.to(device) for t in batch)
            if model_type == "bert":
                loss = model(*batch)[0] # input_ids, input_mask, segment_ids, masked_lm_labels=lm_label_ids, next_sentence_label=is_next)[0]
            elif model_type == "xlnet":
                loss = model(batch) # input_ids, input_mask, segment_ids, labels=lm_label_ids)[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_bar.set_postfix({
                "loss": loss.item()
            })
            iteration_end_callback({
                "loss": loss.item(),
                "batch_index": i,
                "epoch": epoch,
                "type": "train"
            })

        val_loss = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="val")
            for i, batch in enumerate(val_bar):
                batch = all_to_device(batch, device)
                # input_ids, input_mask, segment_ids, lm_label_ids, is_next = tuple(t.to(device) for t in batch)
                if model_type == "bert":
                    loss = model(*batch)[0] # input_ids, input_mask, segment_ids, masked_lm_labels=lm_label_ids, next_sentence_label=is_next)[0]
                elif model_type == "xlnet":
                    loss = model(batch) # input_ids, input_mask, segment_ids, labels=lm_label_ids)[0]

                val_bar.set_postfix({
                    "loss": loss.item()
                })
                val_loss += loss.item()

                iteration_end_callback({
                    "loss": loss.item(),
                    "batch_index": i,
                    "epoch": epoch,
                    "type": "validation"
                })

        epoch_end_callback(model, {
            "loss": val_loss / len(val_loader)
        })



class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob > 0.9 and getattr(tokenizer, "vocab", None):
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.extend(tokenizer.convert_tokens_to_ids([token]))
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.extend(tokenizer.convert_tokens_to_ids(["[UNK]"]))
                print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_b, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if example.guid < 5:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("LM label: %s " % (lm_label_ids))
        print("Is next sentence label: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next)
    return features


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        #self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        # load samples into memory
        if on_memory:
            pickle_path = ".".join([corpus_path, "bert.features.pt"])
            try:
                with open(pickle_path, "rb") as f:
                    self.corpus_lines, self.all_docs, self.sample_to_doc = pickle.load(f)
                    print("Loaded BERT features from cache")
            except IOError:
                self.all_docs = []
                doc = []
                self.corpus_lines = 0
                with open(corpus_path, "r", encoding=encoding) as f:
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        line = line.strip()
                        tokens = self.tokenizer.tokenize(line)
                        if not tokens and len(doc):
                            self.all_docs.append(doc)
                            doc = []
                            #remove last added sample because there won't be a subsequent line anymore in the doc
                            self.sample_to_doc.pop()
                        elif len(line):
                            #store as one sample
                            sample = {"doc_id": len(self.all_docs),
                                      "line": len(doc)}
                            self.sample_to_doc.append(sample)
                            doc.append(line)
                            self.corpus_lines = self.corpus_lines + 1

                # if last row in file is not empty
                if self.all_docs[-1] != doc and len(doc):
                    self.all_docs.append(doc)
                    self.sample_to_doc.pop()

                with open(pickle_path, "wb") as pf:
                    pickle.dump((self.corpus_lines, self.all_docs, self.sample_to_doc), pf, protocol=pickle.HIGHEST_PROTOCOL)

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        t1, t2, is_next_label = self.random_sent(item)

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        tokens_b = self.tokenizer.tokenize(t2)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next))

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, label

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"]+1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1, t2
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "" :
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                t2 = next(self.file).strip()
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t2 == "" or t1 == "":
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
                    self.current_doc = self.current_doc+1
            self.line_buffer = t2

        assert t1 != ""
        assert t2 != ""
        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs)-1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                #pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            #check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


special_symbols = {
    "[UNK]"  : 0,
    "[CLS]"  : 1,
    "[SEP]"  : 2,
    "[PAD]"  : 3,
    "[MASK]" : 4,
}
UNK_ID = special_symbols["[UNK]"]
CLS_ID = special_symbols["[CLS]"]
SEP_ID = special_symbols["[SEP]"]
MASK_ID = special_symbols["[MASK]"]


def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
    """Split two segments from `data` starting from the index `begin_idx`."""

    data_len = data.shape[0]
    if begin_idx + tot_len >= data_len:
        print("[_split_a_and_b] returns None: "
                "begin_idx %d + tot_len %d >= data_len %d",
                begin_idx, tot_len, data_len)
        return None

    end_idx = begin_idx + 1
    cut_points = []
    while end_idx < data_len:
        if sent_ids[end_idx] != sent_ids[end_idx - 1]:
            if end_idx - begin_idx >= tot_len: break
            cut_points.append(end_idx)
        end_idx += 1

    a_begin = begin_idx
    if len(cut_points) == 0 or random.random() < 0.5:
        # NotNext
        label = 0
        if len(cut_points) == 0:
            a_end = end_idx
        else:
            a_end = random.choice(cut_points)

        b_len = max(1, tot_len - (a_end - a_begin))
        # (zihang): `data_len - 1` to account for extend_target
        b_begin = random.randint(0, data_len - 1 - b_len)
        b_end = b_begin + b_len
        while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
            b_begin -= 1
        # (zihang): `data_len - 1` to account for extend_target
        while b_end < data_len - 1 and sent_ids[b_end - 1] == sent_ids[b_end]:
            b_end += 1

        new_begin = a_end
    else:
        # isNext
        label = 1
        a_end = random.choice(cut_points)
        b_begin = a_end
        b_end = end_idx

        new_begin = b_end

    while a_end - a_begin + b_end - b_begin > tot_len:
        if a_end - a_begin > b_end - b_begin:
            # delete the right side only for the LM objective
            a_end -= 1
        else:
            b_end -= 1

    ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

    if extend_target:
        if a_end >= data_len or b_end >= data_len:
            print("[_split_a_and_b] returns None: "
                          "a_end %d or b_end %d >= data_len %d",
                          a_end, b_end, data_len)
            return None
        a_target = data[a_begin + 1: a_end + 1]
        b_target = data[b_begin: b_end + 1]
        ret.extend([a_target, b_target])

    return ret

def _is_start_piece(piece):
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    piece = ''.join(piece)
    if (piece.startswith("▁") or piece.startswith("<")
        or piece in special_pieces):
        return True
    else:
        return False

def _sample_mask(sp, seg, mask_alpha, mask_beta,
                 reverse=False, max_gram=5, goal_num_predict=None):
    """Sample `goal_num_predict` tokens for partial prediction.
    About `mask_beta` tokens are chosen in a context of `mask_alpha` tokens."""

    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)

    num_predict = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True)

    if reverse:
        seg = np.flip(seg, 0)

    cur_len = 0
    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict: break

        n = np.random.choice(ngrams, p=pvals)
        if goal_num_predict is not None:
            n = min(n, goal_num_predict - num_predict)
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        # Find the start position of a complete token
        beg = cur_len + l_ctx
        while beg < seg_len and not _is_start_piece(sp.convert_ids_to_tokens([seg[beg].item()])):
            beg += 1
        if beg >= seg_len:
            break

        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_start_piece(sp.convert_ids_to_tokens([seg[beg].item()])):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        # Update
        mask[beg:end] = True
        num_predict += end - beg

        cur_len = end + r_ctx

    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1

    if reverse:
        mask = np.flip(mask, 0)

    return mask


def _create_data(sp, input_paths, seq_len, reuse_len,
                bi_data, num_predict, mask_alpha, mask_beta):
    features = []

    pickle_name = ".".join([input_paths, "xlnet.features.pt"])

    try:
        with open(pickle_name, "rb") as f:
            print("Read features from cache")
            return pickle.load(f)
    except IOError:
        pass

    f = open(input_paths, 'r')
    lines = f.readlines()
    input_data, sent_ids, sent_id = [], [], True

    for line in tqdm(lines, desc="tokenizing"):
        tokens = sp.tokenize(line)
        cur_sent = sp.convert_tokens_to_ids(tokens)
        input_data.extend(cur_sent)
        sent_ids.extend([sent_id] * len(cur_sent))
        sent_id = not sent_id

    # shape of data : [1, 582]
    data = np.array([input_data], dtype=np.int64)
    sent_ids = np.array([sent_ids], dtype=np.bool)

    assert reuse_len < seq_len - 3

    data_len = data.shape[1]
    sep_array = np.array([SEP_ID], dtype=np.int64)
    cls_array = np.array([CLS_ID], dtype=np.int64)

    progress = tqdm(desc="featurizing", total=data_len)

    i = 0
    while i + seq_len <= data_len:
        inp = data[0, i: i + reuse_len]
        tgt = data[0, i + 1: i + reuse_len + 1]

        results = _split_a_and_b(
            data[0], # all line in one Text file.
            sent_ids[0],
            begin_idx=i + reuse_len,
            tot_len=seq_len - reuse_len - 3,
            extend_target=True)

        if not results:
            i += reuse_len
            progress.update(reuse_len)
            continue

        # unpack the results
        (a_data, b_data, label, _, a_target, b_target) = tuple(results)

        # sample ngram spans to predict
        reverse = bi_data
        if num_predict is None:
            num_predict_0 = num_predict_1 = None
        else:
            num_predict_1 = num_predict // 2
            num_predict_0 = num_predict - num_predict_1

        mask_0 = _sample_mask(sp, inp, mask_alpha, mask_beta, reverse=reverse,
                              goal_num_predict=num_predict_0)
        mask_1 = _sample_mask(sp, np.concatenate([a_data, sep_array, b_data,
                                                  sep_array, cls_array]),
                              mask_alpha, mask_beta,
                              reverse=reverse, goal_num_predict=num_predict_1)

        # concatenate data
        cat_data = np.concatenate([inp, a_data, sep_array, b_data,
                                   sep_array, cls_array])
        seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] +
                  [1] * b_data.shape[0] + [1] + [2])
        assert cat_data.shape[0] == seq_len
        assert mask_0.shape[0] == seq_len // 2
        assert mask_1.shape[0] == seq_len // 2

        # the last two CLS's are not used, just for padding purposes
        tgt = np.concatenate([tgt, a_target, b_target, cls_array, cls_array])
        assert tgt.shape[0] == seq_len

        is_masked = np.concatenate([mask_0, mask_1], 0).ravel()
        if num_predict is not None:
            assert np.sum(is_masked) == num_predict

        # print(is_masked.shape)

        feature = {
            "input": cat_data,
            "is_masked": is_masked,
            "target": tgt,
            "seg_id": seg_id,
            "label": [label],
        }
        features.append(feature)

        i += reuse_len
        progress.update(reuse_len)

    f.close()

    #with open(pickle_name, "wb") as pf:
    #    pickle.dump(features, pf)

    return features


class XLNetDataset(Dataset):
    """A Dataset with xlnet features created from a corpus."""
    def __init__(self, tokenizer, path, seq_len, reuse_len, bi_data, num_predict, mask_alpha, mask_beta):
        super().__init__()
        self.features = _create_data(tokenizer, path, seq_len, reuse_len, bi_data, num_predict, mask_alpha, mask_beta)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
    """
    Sample a permutation of the factorization order, and create an
    attention mask accordingly.
    Args:
    inputs: int64 Tensor in shape [seq_len], input ids.
    targets: int64 Tensor in shape [seq_len], target ids.
    is_masked: bool Tensor in shape [seq_len]. True means being selected
      for partial prediction.
    perm_size: the length of longest permutation. Could be set to be reuse_len.
      Should not be larger than reuse_len or there will be data leaks.
    seq_len: int, sequence length.
    """

    # Generate permutation indices
    index = torch.arange(seq_len, dtype=torch.int64)

    index = torch.reshape(index, [-1, perm_size]).t()
    index = index[torch.randperm(index.shape[0])]
    index = torch.reshape(index.t(), [-1])

    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens = ~(torch.eq(inputs, SEP_ID) | torch.eq(inputs, CLS_ID))
    non_mask_tokens = (~is_masked) & non_func_tokens
    masked_or_func_tokens = ~non_mask_tokens

    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    smallest_index = -torch.ones([seq_len], dtype=torch.int64)

    # put -1 if `non_mask_tokens(real token not cls or sep)` not permutation index
    # print(non_mask_tokens.shape, smallest_index.shape, index.shape, seq_len)
    rev_index = torch.where(non_mask_tokens.cpu(), smallest_index.cpu(), index.cpu())

    # Create `target_mask`: non-funcional and maksed tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_tokens = masked_or_func_tokens & non_func_tokens
    target_mask = target_tokens.type(torch.float32)

    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    # put `rev_index` if real mask(not cls or sep) else `rev_index + 1`
    self_rev_index = torch.where(target_tokens.cpu(), rev_index.cpu(), rev_index.cpu() + 1)

    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) &  masked_or_func_tokens.cpu()
    perm_mask = perm_mask.type(torch.float32).to(inputs.device)

    # new target: [next token] for LM and [curr token] (self) for PLM
    new_targets = torch.cat([inputs[0: 1], targets[: -1]], dim=0)

    # construct inputs_k
    inputs_k = inputs

    # construct inputs_q
    inputs_q = target_mask

    return perm_mask, new_targets, target_mask, inputs_k, inputs_q


def make_permute(feature, reuse_len, seq_len, perm_size, num_predict):
    inputs = feature.pop("input").long()
    target = feature.pop("target").long()
    is_masked = feature.pop("is_masked").byte()

    non_reuse_len = seq_len - reuse_len
    assert perm_size <= reuse_len and perm_size <= non_reuse_len

    perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0 = _local_perm(
        inputs[:reuse_len], # inp
        target[:reuse_len],
        is_masked[:reuse_len],
        perm_size,
        reuse_len)

    perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1 = _local_perm(
        inputs[reuse_len:], # (senA, seq, senBm seq, cls)
        target[reuse_len:],
        is_masked[reuse_len:],
        perm_size,
        non_reuse_len)

    perm_mask_0 = torch.cat([perm_mask_0, torch.ones([reuse_len, non_reuse_len]).to(perm_mask_0.device)],
                            dim=1)
    perm_mask_1 = torch.cat([torch.zeros([non_reuse_len, reuse_len]).to(perm_mask_1.device), perm_mask_1],
                            dim=1)

    perm_mask = torch.cat([perm_mask_0, perm_mask_1], dim=0)
    target = torch.cat([target_0, target_1], dim=0)
    target_mask = torch.cat([target_mask_0, target_mask_1], dim=0)
    input_k = torch.cat([input_k_0, input_k_1], dim=0)
    input_q = torch.cat([input_q_0, input_q_1], dim=0)

    if num_predict is not None:
        indices = torch.arange(seq_len, dtype=torch.int64)
        bool_target_mask = target_mask.byte()
        indices = indices[bool_target_mask]

        ##### extra padding due to CLS/SEP introduced after prepro
        actual_num_predict = indices.shape[0]
        pad_len = num_predict - actual_num_predict

        assert seq_len >= actual_num_predict

        ##### target_mapping
        target_mapping = torch.eye(seq_len, dtype=torch.float32)[indices]
        paddings = torch.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
        target_mapping = torch.cat([target_mapping, paddings], dim=0)
        feature["target_mapping"] = torch.reshape(target_mapping,
                                                [num_predict, seq_len]).to(inputs.device)
        ##### target
        target = target[bool_target_mask]
        paddings = torch.zeros([pad_len], dtype=target.dtype, device=target.device)
        target = torch.cat([target, paddings], dim=0)
        feature["target"] = torch.reshape(target, [num_predict])

        ##### target mask
        target_mask = torch.cat(
            [torch.ones([actual_num_predict], dtype=torch.float32),
             torch.zeros([pad_len], dtype=torch.float32)],
            dim=0).to(inputs.device)
        feature["target_mask"] = torch.reshape(target_mask, [num_predict])
    else:
        feature["target"] = torch.reshape(target, [seq_len])
        feature["target_mask"] = torch.reshape(target_mask, [seq_len])

    # reshape back to fixed shape
    feature["seg_id"] = torch.stack(feature["seg_id"], dim=0).long()
    feature["perm_mask"] = torch.reshape(perm_mask, [seq_len, seq_len])
    feature["input_k"] = torch.reshape(input_k, [seq_len])
    feature["input_q"] = torch.reshape(input_q, [seq_len])

    return feature


def all_to_device(collection, device):
    if isinstance(collection, torch.Tensor):
        return collection.to(device)
    elif isinstance(collection, dict):
        return {k: all_to_device(v, device) for k, v in collection.items()}
    else:
        return [all_to_device(v, device) for v in collection]


def save_model_on_better_loss(output_directory):
    best_loss = float("inf")

    def save_model(model, statistics):
        nonlocal best_loss

        if statistics["loss"] < best_loss:
            with open(os.path.join(output_directory, "model.pt"), "wb") as f:
                torch.save(model.state_dict(), f)

            best_loss = statistics["loss"]

    return save_model


def log_statistics(output_directory):
    statistics_file = open(os.path.join(output_directory, "statistics"), "wt")

    def on_iteration(statistics):
        statistics_file.write(json.dumps(statistics) + "\n")

    return on_iteration


def recursive_squeeze_batch(value):
    if isinstance(value, torch.Tensor):
        return value.squeeze(0)
    elif isinstance(value, dict):
        return {
            k: recursive_squeeze_batch(v) for k, v in value.items()
        }
    else:
        return [recursive_squeeze_batch(v) for v in value]


class XLNetModelWithPermutations(nn.Module):
    """Implement XLNet, but do permutations as well."""
    def __init__(self, config):
        super().__init__()
        self.xlnet_head = XLNetLMHeadModel(config)

    def forward(self, feature):
        # Batch size should be 1, otherwise this won't work
        # for key in feature:
        #     assert feature[key].shape[0] == 1

        feature = recursive_squeeze_batch(feature)

        permutation = make_permute(
            feature,
            reuse_len=64,
            seq_len=128,
            perm_size=64,
            num_predict=85
        )

        # batch size is 1
        inp_k = permutation['input_k'].unsqueeze(0) # [seq_len, 1(=bsz)]
        seg_id = permutation['seg_id'].unsqueeze(0) # [seq_len, 1(=bsz)]
        target = permutation['target'].unsqueeze(0) # [num_predict, 1(=bsz)]
        perm_mask = permutation['perm_mask'].unsqueeze(0) # [seq_len, seq_len, 1(=bsz)]
        target_mapping = \
            permutation['target_mapping'].unsqueeze(0) # [num_predict, seq_len, 1(=bsz)]
        inp_q = permutation['input_q'].unsqueeze(0) # [seq_len, 1(=bsz)]
        tgt_mask = permutation['target_mask'].unsqueeze(0) # [num_predict, 1(=bsz)]

        loss = self.xlnet_head(inp_k,
                               token_type_ids=seg_id,
                               input_mask=None,
                               attention_mask=None,
                               mems=None,
                               perm_mask=perm_mask,
                               target_mapping=target_mapping,
                               labels=target)[0]

        return loss


def model_and_tokenizer(model_type, do_lower_case=True):
    """Get the model and tokenizer."""
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                do_lower_case=True)
        config = BertConfig(
            vocab_size_or_config_json_file=len(list(tokenizer.vocab.keys())),
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02
        )
        model = BertForPreTraining(config)
    elif model_type == "xlnet":
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased",
                                                   do_lower_case=True)
        config = XLNetConfig.from_pretrained("xlnet-base-cased")
        model = XLNetModelWithPermutations(config)

    return tokenizer, config, model


def datasets(model_type, tokenizer, train_corpus, val_corpus, seq_len):
    if model_type == "bert":
        return (
            BERTDataset(train_corpus, tokenizer, seq_len=seq_len),
            BERTDataset(val_corpus, tokenizer, seq_len=seq_len)
        )
    elif model_type == "xlnet":
        return (
            XLNetDataset(tokenizer,
                         train_corpus,
                         seq_len,
                         64,
                         False,
                         85,
                         6,
                         1),
            XLNetDataset(tokenizer,
                         val_corpus,
                         seq_len,
                         64,
                         False,
                         85,
                         6,
                         1)
       )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_corpus")
    parser.add_argument("val_corpus")
    parser.add_argument("--output-directory")
    parser.add_argument("--max-seq-length", default=128, type=int)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--adam-epsilon", default=1e-8)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--grad-accum-steps", default=1)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--model-type", choices=("bert", "xlnet"), type=str)

    args = parser.parse_args()
    args.batch_size = args.batch_size // args.grad_accum_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_directory, exist_ok=True)

    tokenizer, config, model = model_and_tokenizer(args.model_type)
    model = model.to(args.device)

    train_dataset, val_dataset = datasets(args.model_type,
                                          tokenizer,
                                          args.train_corpus,
                                          args.val_corpus,
                                          args.max_seq_length)

    #train_dataset = BERTDataset(args.train_corpus, tokenizer, seq_len=args.max_seq_length)
    #val_dataset = BERTDataset(args.val_corpus, tokenizer, seq_len=args.max_seq_length)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size
    )
    train_optimization_steps = int(
        len(train_dataset) / args.batch_size / args.grad_accum_steps
    ) * args.epochs

    params_for_optimization = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_params = [
        {"params": [p for n, p in params_for_optimization if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in params_for_optimization if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_params, lr=args.lr, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.warmup,
                                     t_total=train_optimization_steps)

    train_loop(model,
               optimizer,
               scheduler,
               train_loader,
               val_loader,
               args.epochs,
               args.device,
               args.model_type,
               epoch_end_callback=save_model_on_better_loss(args.output_directory),
               iteration_end_callback=log_statistics(args.output_directory))


if __name__ == "__main__":
    main()