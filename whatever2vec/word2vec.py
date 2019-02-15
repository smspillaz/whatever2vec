import argparse
import gensim
from gensim.models import KeyedVectors

from .utils import EpochLogging, yield_sentences


def train(documents, size=150, window=10, save=None):
    model = gensim.models.Word2Vec(
        documents,
        size=size,
        window=window,
        min_count=2,
        workers=10,
        callbacks=[EpochLogging()]
    )
    model.train(documents, total_examples=len(documents), epochs=10)

    if save:
        model.save(save)

    return model.wv


def main():
    parser = argparse.ArgumentParser("Train node2word2vec")
    parser.add_argument("--train", type=str, help="The training sentences file")
    parser.add_argument("--valid", type=str, help="The validation sentences file")
    parser.add_argument("--test", type=str, help="The test sentences file")
    parser.add_argument("--save", type=str, help="Where to save the model to")
    parser.add_argument("--load", type=str, help="Where to load the model from")
    args = parser.parse_args()

    if args.load:
        vectors = KeyedVectors.load(args.load, mmap='r')
    else:
        vectors = train(list(yield_sentences(args.train)), save=args.save)


if __name__ == "__main__":
    main()
