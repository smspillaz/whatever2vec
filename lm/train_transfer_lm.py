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
        model = XLNetLMHeadModel(config)
    
    return tokenizer, config, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_corpus")
    parser.add_argument("val_corpus")
    parser.add_argument("--output-directory")
    parser.add_argument("--max-seq-length", default=128, type=int)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--adam-epsilon", default=1e-8)
    parser.add_argument("--epochs", default=3)
    parser.add_argument("--warmup", default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", default=42)
    parser.add_argument("--grad-accum-steps", default=1)
    parser.add_argument("--batch-size", default=32)
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

    train_dataset = BERTDataset(args.train_corpus, tokenizer, seq_len=args.max_seq_length)
    val_dataset = BERTDataset(args.val_corpus, tokenizer, seq_len=args.max_seq_length)
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