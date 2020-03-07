#!/usr/bin/env python
"""Convert PyTorch AWD-Language Model Embeddings to word vectors."""

import argparse
import os
import json
import torch

import pytorch_transformers

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab


def load_dictionary_pickle(pickle_file):
    return torch.load(pickle_file).dictionary.idx2word


def load_dictionary_txtlist(path):
    with open(path, "r") as f:
        return dict([(i, x.strip()) for i, x in enumerate(f)])


def load_dictionary_json(path):
    with open(path, "r") as f:
        return {
            v: k for k, v in json.load(f).items()
        }


def get_one_of(object, attribs):
    """Try to get the first one of attribs."""
    for a in attribs:
        v = getattr(object, a, None)
        if v:
            return v


def load_dictionary_auto(path):
    tokenizer = pytorch_transformers.AutoTokenizer.from_pretrained(path)
    return {
        v: k for k, v in get_one_of(tokenizer, ['vocab', 'encoder']).items()
    }


LOAD_DICTIONARY_DISPATCH = {
    ".pt": load_dictionary_pickle,
    ".txt": load_dictionary_txtlist,
    ".json": load_dictionary_json,
    "": load_dictionary_auto
}


def load_dictionary(filename):
    """Load dictionary from filename."""
    path, ext = os.path.splitext(filename)
    return LOAD_DICTIONARY_DISPATCH[ext](filename)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("PyTorch Language Model Embeddings to Word Vectors")
    parser.add_argument("--model", required=True)
    parser.add_argument("--embeddings-layer", required=True, type=str)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    dictionary = load_dictionary(args.dictionary)

    try:
        model = torch.load(args.model, map_location='cpu')
    except IOError:
        model = pytorch_transformers.AutoModelWithLMHead.from_pretrained(args.model).state_dict()
    except ValueError:
        raise RuntimeError("Can't load model {}, either not on disk, "
                           "or not such pretrained model exists".format(args.model))

    try:
        embeddings = model[args.embeddings_layer].cpu().numpy()
    except KeyError:
        raise RuntimeError("Couldn't get {}, keys are {}".format(args.embeddings_layer, model.keys()))

    if args.transpose:
        embeddings = embeddings.T

    kv = KeyedVectors(embeddings.shape[1])
    kv.syn0 = embeddings
    kv.vocab = {
        w: Vocab(index=i) for i, w in dictionary.items()
    }
    kv.index2word = dictionary

    kv.save(args.output)


if __name__ == "__main__":
    main()
