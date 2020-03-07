import argparse
import gc
import json
import pickle
import numpy as np
import os
import torch

import pytorch_transformers

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab

from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm


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


class DictionaryDataset(Dataset):
    """Define a dataset which iterates over all the words in the dictionary."""
    def __init__(self, dictionary, tokenizer):
        """Initialize."""
        super().__init__()
        self.dictionary_tokens = list(dictionary.values())
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dictionary_tokens)

    def __getitem__(self, i):
        tokenized_text = self.tokenizer.add_special_tokens_single_sentence(
            self.tokenizer.convert_tokens_to_ids([
                    self.dictionary_tokens[i]
            ])
        )
        return torch.tensor(tokenized_text)


def yield_embeddings_chunks(loader, model, device=None):
    device = device or "cpu"

    with torch.no_grad():
        for examples in loader:
            examples = examples.to(device)

            encoded, _, hiddens = model(examples)
            #tqdm.write("{}".format(len(hiddens)))
            #tqdm.write("{}".format([h.shape for h in hiddens]))
            hiddens = torch.cat([h[:, 1, :].unsqueeze(1) for h in hiddens], dim=1)
            #tqdm.write("{}".format(hiddens.shape))
            yield hiddens.cpu()


def configure_model(model, model_type):
    if model_type.startswith("gpt2"):
        model.output_hidden_states = True
    elif "bert" in model_type:
        model.encoder.output_hidden_states = True

    return model


def main():
    """Entry point, the place where we load the model and dictionaries etc."""
    parser = argparse.ArgumentParser("PyTorch Transformer Embedding to Word Vectors")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", default=1, type=int)
    args = parser.parse_args()

    dictionary = load_dictionary(args.dictionary)

    try:
        with open(args.weights, "r") as f:
            state_dict = torch.load(f, map_location='cpu')
    except (TypeError, IOError):
        state_dict = pytorch_transformers.AutoModel.from_pretrained(args.model).state_dict()
    except ValueError:
        raise RuntimeError("Can't load model {}, either not on disk, "
                           "or not such pretrained model exists".format(args.model))

    tokenizer = pytorch_transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    model = pytorch_transformers.AutoModel.from_pretrained(args.model).to(args.device)
    model.load_state_dict(state_dict)
    model = configure_model(model, args.model)

    # Now that we have the model, we'll need to take every word in the dictionary and
    # run it through the model, with the special tokens in order to get the encodings.
    # This is straightforward enough, though we just need to run without any shuffling
    dataset = DictionaryDataset(dictionary, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model.eval()

    # Now do the loop
    all_embeddings = torch.cat([
        embeddings for embeddings in yield_embeddings_chunks(
            tqdm(loader, desc="Extracting layer embeddings chunks"),
            model,
            device=args.device
        )
    ], dim=0)
    print(all_embeddings.shape)

    embeddings_per_layer = [
        e.squeeze(1).numpy() for e in torch.split(all_embeddings, 1, dim=1)
    ]
    print([e.shape for e in embeddings_per_layer])

    del all_embeddings
    gc.collect()

    for i, embeddings in enumerate(embeddings_per_layer):
        kv = KeyedVectors(embeddings.shape[1])
        kv.syn0 = embeddings
        kv.vocab = {
            w: Vocab(index=idx) for idx, w in dictionary.items()
        }
        kv.index2word = dictionary
        kv.save(".".join([args.output, "layer", str(i), "wv"]), pickle_protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

