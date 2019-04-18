import itertools
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS

from utils import EpochLogging, yield_sentences, clean_terms, options


def get_text_for_gow(file, min_num_tokens=6, stopwords=STOPWORDS, lemmatize=None, stem=None, only_tags=None):
    """
    Get list of lists of tokens from each document by applying preprocessing.
    :param file: path to the dataset
    :param min_num_tokens: discard every list of tokens (document) with length < this
    :param stopwords: list of stopwords
    :param lemmatize: boolean, whether you want to lemmatize the terms or not
    :param stem: boolean, whether you want to stem the terms or not
    :param only_tags: list of tags you want to keep, for example only nouns (NN)
    :return:
    """
    documents = list(yield_sentences(file, min_num_tokens=min_num_tokens, subsample=None))
    if stopwords or lemmatize or stem or only_tags:
        documents = [clean_terms(doc, stopwords, lemmatize, stem, only_tags) for doc in documents]
    return documents


def terms_to_graph(documents, w, weight_type='co-occurrences'):
    """
    Constructs a dictionary of tuples/edges -> weights
    :param documents: list of list of tokens
    :param w: window size
    :param weight_type: ('co-occurrences', 'inverse_distance', 'cosine_similarity')
    :return: dictionary of edge data
    """
    from_to = {}

    for terms in documents:
        w = min(w, len(terms))
        # create initial graph (first w terms)
        terms_temp = terms[0:w]
        indexes = list(itertools.combinations(range(w), r=2))

        new_edges = list()

        for i in range(len(indexes)):
            new_edges.append(" ".join(list(terms_temp[i] for i in indexes[i])))
        for i in range(0, len(new_edges)):
            from_to[new_edges[i].split()[0], new_edges[i].split()[1]] = 1

        # then iterate over the remaining terms
        for i in range(w, len(terms)):
            # term to consider
            considered_term = terms[i]
            # all terms within sliding window
            terms_temp = terms[(i - w + 1):(i + 1)]

            # edges to try
            candidate_edges = list()
            for p in range(w - 1):
                candidate_edges.append((terms_temp[p], considered_term))

            for try_edge in candidate_edges:
                # if not self-edge
                if try_edge[1] != try_edge[0]:
                    boolean1 = (try_edge[0], try_edge[1]) in from_to
                    boolean2 = (try_edge[1], try_edge[0]) in from_to
                    # if edge has already been seen, update its weight
                    if boolean1:
                        from_to[try_edge[0], try_edge[1]] += 1
                    elif boolean2:
                        from_to[try_edge[1], try_edge[0]] += 1
                    # if edge has never been seen, create it and assign it a unit weight
                    else:
                        from_to[try_edge] = 1
    return from_to


def train(G, dimensions=50, walk_length=30, num_walks=200, workers=10, temp_folder='node2vec_temp', save=None):
    """
    Trains the node2vec model on the given graph
    :param G: the graph
    :param dimensions: embeddings dimension
    :param walk_length: the length of the random walks
    :param num_walks: the number of the random walks
    :param workers: number of threads (for Windows it only works with 1 worker)
    :param temp_folder: folder to store temp data during the parallel process when the graph is big
    :param save: filepath to save the embeddings in word2vec format
    :return:
    """
    # Precompute probabilities and generate walks
    node2vec = Node2Vec(
        graph=G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        temp_folder=temp_folder
    )
    # Embed nodes
    # Any keywords acceptable by gensim.Word2Vec can be passed.
    # `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    model = node2vec.fit(
        window=10,
        min_count=2,
        workers=workers,
        callbacks=[EpochLogging()]
    )
    if save:
        # Save model embeddings for later use
        model.wv.save_word2vec_format(save)
        #model.save(save)
    return model.wv


def dict_to_networkx(g_dict, name=None):
    """
    Transforms the edge data dictionary to NetworkX graph
    :param g_dict: edge data dictionary
    :param name: name of the graph
    :return: nx.Graph
    """
    G = nx.Graph()
    G.name = name
    for edge, weight in g_dict.items():
        G.add_edge(*edge, weight=weight)
    print(G.number_of_nodes())
    print(G.number_of_edges())
    return G


def plot_degree_histogram(G, log_yscale=True):
    """
    Plots the histogram of the degree
    :param G: nx.Graph
    :param log_yscale: if True, pyplot's yscale is log
    """
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    plt.hist(degree_sequence)
    if log_yscale:
        plt.yscale('log')
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.show()


def main():
    args = options('Train word2node2vec')

    if args.load:
        vectors = KeyedVectors.load(args.load, mmap='r')
    else:
        documents = get_text_for_gow(args.train, lemmatize=None)
        gow_dict = terms_to_graph(documents, w=10)
        G = dict_to_networkx(gow_dict)
        # plot_degree_histogram(G)
        vectors = train(G, save=args.save, dimensions=50)


if __name__ == "__main__":
    main()
