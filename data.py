
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def encode_data(data, tokenizer, punctuation_enc):
    """
    Converts words to (BERT) tokens and puntuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X = []
    Y = []
    for line in data:
        word, punc = line.split('\t')
        punc = punc.strip()
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        y = [punctuation_enc[punc]]
        if len(x) > 0:
            if len(x) > 1:
                y = (len(x)-1)*[0]+y
            X += x
            Y += y
    return X, Y

def insert_target(x, segment_size):
    """
    Creates segments of surrounding words for each word in x.
    Inserts a zero token halfway the segment.
    """
    X = []
    x_pad = x[-((segment_size-1)//2-1):]+x+x[:segment_size//2]

    for i in range(len(x_pad)-segment_size+2):
        segment = x_pad[i:i+segment_size-1]
        segment.insert((segment_size-1)//2, 0)
        X.append(segment)

    return np.array(X)

def preprocess_data(data, tokenizer, punctuation_enc, segment_size):
    X, y = encode_data(data, tokenizer, punctuation_enc)
    X = insert_target(X, segment_size)
    return X, y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader