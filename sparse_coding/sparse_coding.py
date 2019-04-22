import argparse
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
import time
from datetime import timedelta
from gensim.models import KeyedVectors
from sklearn.decomposition import MiniBatchDictionaryLearning

IS_TRAINING = True
IS_ANALYSIS = True
EMBEDDING_SIZE = 50
VOCAB_SIZE = 10000
# COMPONENT_SIZE = 2000
COMPONENT_SIZE = 500
test_words = ["bank", "bass", "tie", "chips", "mouse", "crane", "spring",
              "light", "star", "helsinki", "finland"]

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--model', default='', type=str,
                    choices=('word2node2vec', 'word2vec', 'lm2vec'),
                    help='Type of word embedding model to load')
args = parser.parse_args()
DICTIONARY_FILENAME = f"dict_{args.model}_{EMBEDDING_SIZE}_vocab={VOCAB_SIZE}_components={COMPONENT_SIZE}.model"


def sample_embeddings(model_wv, words=[], restrict_vocab=10000):
    embeddings = pd.DataFrame(model_wv.wv.syn0[:restrict_vocab])
    for i, word in enumerate(words):
        embedding = model_wv.get_vector(word)
        # Just overwrite
        embeddings.iloc[i] = embedding
    return embeddings

def get_sparse_code(dictionary, model_wv, word):
    """Get the activated atoms and sparse codes for a word."""
    embedding = model_wv.get_vector(word)  # (50, )
    embedding = embedding.reshape(1, -1)  # (1, 50)
    sparse_code = dictionary.transform(embedding)  # (1, 2000)
    sparse_code = sparse_code.flatten()  # (2000, )
    atom_indices = np.where(sparse_code > 0)
    print("\n\tAtoms: " + str(atoms_indices))
    # The activated atoms' embddings are the dictionary's basis vectors
    # with non-zero coefficients
    atom_embeddings = dictionary.components_[sparse_code > 0]
    return atom_indices, atom_embeddings


# Load Word2VecKeyedVectors object
if args.model == 'word2node2vec':
    # model_wv = KeyedVectors.load_word2vec_format('../whatever2vec/w2n2v.embeddings', binary=False)
    model_wv = KeyedVectors.load_word2vec_format('../whatever2vec/w2n2v.embeddings_new', binary=False)
elif args.model == 'word2vec':
    model_wv = KeyedVectors.load(f'../models/w2v/vectors/vectors.{EMBEDDING_SIZE}.vw')
elif args.model == 'lm':
    model_wv = KeyedVectors.load(f'../models/w2v/vectors/lm/WT103.24h.QRNN.{EMBEDDING_SIZE}.vw')

# Only sample N word embeddings
selected_df = sample_embeddings(model_wv,
                                words=test_words,
                                restrict_vocab=VOCAB_SIZE)
print(selected_df.shape)

if IS_TRAINING:
    # Train dictionary learning model
    num_cpus = multiprocessing.cpu_count()
    X = selected_df.values
    dictionary = MiniBatchDictionaryLearning(n_components=COMPONENT_SIZE, fit_algorithm='lars',
                                             transform_algorithm='lars',
                                             transform_n_nonzero_coefs=5, verbose=1,
                                             n_jobs=num_cpus, batch_size=16, n_iter=1000)
    start = time.time()
    dictionary.fit(X)
    elapsed = time.time() - start
    print("Dictionary learning took " + str(timedelta(seconds=elapsed)))
    # Dictionary learning took 0:07:00.324986 on 1000 words
    # Dictionary learning took 2:09:44.087086 on 10000 words LOL
    # Save model
    with open(DICTIONARY_FILENAME, 'wb') as f:
        pickle.dump(dictionary, f)

# Analyze "atoms of discourse"
if IS_ANALYSIS:
    with open(DICTIONARY_FILENAME, 'rb') as f:
        dictionary = pickle.load(f)
    # basis_vectors = dictionary.components_
    # For plotting
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    similarities_df = pd.DataFrame()
    words_df = pd.DataFrame()
    # For vertical demarcations in heatmap
    demarcations = []
    demarcation_i = 0
    word_labels = []
    for word in test_words:
        demarcations.append(demarcation_i)
        print(f"\n\tWord: {word}")
        atom_indices, atom_embeddings = get_sparse_code(dictionary,
                                                        model_wv, word)
        # Get closest words to each atom of discourse
        for i, atom in zip(atom_indices, atom_embeddings):
            word_labels.append(f"{word} ({i})")
            demarcation_i += 1
            similar_words = model_wv.similar_by_vector(atom, topn=10)
            print(similar_words)

            words, similarities = zip(*similar_words)
            similarities_df[f"{word}_{i}"] = similarities
            words_df[f"{word}_{i}"] = words
    demarcations.append(demarcation_i)

    idx = [None, None]
    divisor = demarcations[5]  # Since 11 words in total
    idx[0] = similarities_df.columns[:divisor]
    idx[1] = similarities_df.columns[divisor:]
    word_labels = [word_labels[:divisor], word_labels[divisor:]]
    demarcations = [demarcations[:6], [x - divisor for x in demarcations[5:]]]
    fig, axes = plt.subplots(2, 1, sharey=True)
    for i, ax in enumerate(axes.flat):
        sns.heatmap(similarities_df[idx[i]], ax=ax,
                    annot=words_df[idx[i]], fmt="", cmap=cmap,
                    yticklabels=False,  # Hide y-axis labels
                    cbar=(i==0))
        ax.vlines(demarcations[i], *ax.get_ylim(), colors='deepskyblue', lw=4)
        ax.set_xticklabels(word_labels[i])
        ax.xaxis.tick_top()
        ax.tick_params(left=False, top=False)  # Hide ticks
    plt.title(f"{args.model}_{EMBEDDING_SIZE} with {COMPONENT_SIZE} components",
              y=2.4)
    plt.savefig(f"results/full_tables/{args.model}_{EMBEDDING_SIZE}_components={COMPONENT_SIZE}.png")
    plt.show()
