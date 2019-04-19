import multiprocessing
import pandas as pd
import pickle
import time
from datetime import timedelta
from gensim.models import KeyedVectors
from sklearn.decomposition import MiniBatchDictionaryLearning

TRAIN = True
ANALYZE = True
words = ["bank", "cut", "bass", "tie", "chips", "mouse", "crane"]

def sample_embeddings(model_wv, words=[], restrict_vocab=10000):
    embeddings = pd.DataFrame(model_wv.wv.syn0[:restrict_vocab])
    for i, word in enumerate(words):
        embedding = model_wv.get_vector(word)
        # Just overwrite
        embeddings.iloc[i] = embedding
    return embeddings

# Load Word2VecKeyedVectors object
model_wv = KeyedVectors.load_word2vec_format('../whatever2vec/w2n2v.embeddings', binary=False)

# Only sample N word embeddings
selected_df = sample_embeddings(model_wv,
                                words=words,
                                restrict_vocab=1000)
print(selected_df.shape)

if TRAIN:
    # Train dictionary learning model
    num_cpus = multiprocessing.cpu_count()
    X = selected_df.values
    dictionary = MiniBatchDictionaryLearning(n_components=2000, fit_algorithm='lars',
                                             transform_algorithm='lars',
                                             transform_n_nonzero_coefs=5, verbose=1,
                                             n_jobs=num_cpus, batch_size=16, n_iter=1000)
    start = time.time()
    dictionary.fit(X)
    elapsed = time.time() - start
    print("Dictionary learning took " + str(timedelta(seconds=elapsed)))

    # Save model
    with open('dictionary.model', 'wb') as f:
        pickle.dump(dictionary, f)

# Analyze "atoms of discourse"
if ANALYZE:
    with open("dictionary.model", 'rb') as f:
        dictionary = pickle.load(f)
    basis_vectors = dictionary.components_
    for word in words:
        print(f"\n\tWord: {word}")
        embedding = model_wv.get_vector(word)  # (50, )
        sparse_code = dictionary.transform(embedding.reshape(1, -1))  # (1, 2000)
        sparse_code = sparse_code.flatten()  # (2000, )
        activated_atoms = sparse_code > 0
        print("\n\tAtoms: " + str(np.where(sparse_code > 0)))
        activated_atoms = dictionary.components_[activated_atoms]
        # Get closest words to each atom of discourse
        for i, atom in enumerate(activated_atoms):
            similar_words = model_wv.similar_by_vector(atom, topn=10)
            print(similar_words)
