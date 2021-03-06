import argparse
import gensim
from gensim.models import KeyedVectors

from .utils import EpochLogging, yield_sentences, options


def train(documents, size=150, window=10, save=None, sg=0):
    model = gensim.models.Word2Vec(
        documents,
        size=size,
        window=window,
        min_count=2,
        workers=10,
        sg=sg,
        callbacks=[EpochLogging()]
    )
    model.train(documents, total_examples=len(documents), epochs=10)

    if save:
        model.save(save)

    return model.wv


def main():
    args = options('Train word2vec')

    if args.load:
        vectors = KeyedVectors.load(args.load, mmap='r')
    else:
        vectors = train(list(yield_sentences(args.train)), save=args.save, sg=1 if args.sg else 0)


if __name__ == "__main__":
    main()
