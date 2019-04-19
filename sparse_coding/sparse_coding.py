import multiprocessing
import pandas as pd
import pickle
import time
from datetime import timedelta
from gensim.models import KeyedVectors
from sklearn.decomposition import MiniBatchDictionaryLearning

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
                                words=["bank", "cut", "bass", "tie"],
                                restrict_vocab=1000)
print(selected_df.shape)

# Train dictionary learning model
num_cpus = multiprocessing.cpu_count()
X = selected_df.values
dictionary = MiniBatchDictionaryLearning(n_components=2000, fit_algorithm='lars', transform_algorithm='lars',
                                         transform_n_nonzero_coefs=5, verbose=1, n_jobs=num_cpus, batch_size=16, n_iter=1000)
start = time.time()
dictionary.fit(X)
elapsed = time.time() - start
print("Dictionary learning took " + timedelta(seconds=elapsed))

# Save model
with open('dictionary.model', 'wb') as f:
    pickle.dump(model, f)
