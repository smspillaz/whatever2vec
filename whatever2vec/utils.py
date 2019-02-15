import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.parsing.preprocessing import STOPWORDS
from tqdm.auto import tqdm


class EpochLogging(CallbackAny2Vec):
    def __init__(self):
        super().__init__()
        self.epoch = 0

    def on_epoch_begin(self, model):
        print('Started epoch {}'.format(self.epoch))

    def on_epoch_end(self, model):
        print('Finished epoch {}'.format(self.epoch))
        self.epoch += 1


def yield_sentences(filename, min_num_tokens=0):
    """
    Yields the tokens of every document in a list of lists in lowercase.
    :param filename: the path to the dataset
    :param min_num_tokens: discard every list of tokens (document) with length < this
    :return: list of lists of tokens
    """
    with open(filename, "r") as f:
        for sentence in tqdm(f, desc='Reading training data'):
            terms = gensim.utils.simple_preprocess(sentence)
            if len(terms) >= min_num_tokens:
                yield terms


def clean_terms(terms, stopwords=STOPWORDS, lemmatize=None, stem=None, only_tags=None):
    """
    Cleans a list of words.
    :param terms: list of words/terms
    :param stopwords: list of stopwords
    :param lemmatize: boolean, whether you want to lemmatize the terms or not
    :param stem: boolean, whether you want to stem the terms or not
    :param only_tags: list of tags you want to keep, for example only nouns (NN)
    :return: the clean list of words
    """
    if stopwords:
        terms = [t for t in terms if t not in stopwords]
    if only_tags:
        tagged = nltk.pos_tag(terms)
        terms = [t for t, pos in tagged if pos in tags]
    if lemmatize:
        lem = WordNetLemmatizer()
        terms = [lem.lemmatize(t) for t in terms]
    if stem:
        stem = PorterStemmer()
        terms = [stem.stem(t) for t in terms]
    return terms