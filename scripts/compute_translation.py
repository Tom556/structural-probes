from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import torch
import h5py
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

LAYER_NUM = 12
FEATURE_DIM = 768


def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    '''Aligns tokenized and untokenized sentence given subwords "##" prefixed

    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.

    Args:
      tokenized_sent: a list of strings describing a subword-tokenized sentence
      untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
      A dictionary of type {int: list(int)} mapping each untokenized sentence
      index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and
           tokenized_sent_index < len(tokenized_sent)):
        while (tokenized_sent_index + 1 < len(tokenized_sent) and
               tokenized_sent[tokenized_sent_index + 1].startswith('##')):
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1
    return mapping


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('hdf5_file')
    argp.add_argument('sentences_raw')
    argp.add_argument('translation_npz')
    args = argp.parse_args()

    subword_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    with open(args.sentences_raw, 'r') as in_sents:
        sentences = in_sents.readlines()

    hf = h5py.File(args.hdf5_file, 'r')
    indices = list(hf.keys())

    translation = np.zeros((LAYER_NUM, FEATURE_DIM))

    for index, sent in tqdm(zip(sorted([int(x) for x in indices]), sentences), desc='Averaging sentence vectors'):
        sent = sent.strip()
        tokenized_sent = subword_tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + sent + ' [SEP]')
        untokenized_sent = sent.split(' ')
        untok_tok_mapping = match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)
        feature_stack = hf[str(index)]

        for layer_idx in range(LAYER_NUM):
            single_layer_features = feature_stack[layer_idx]
            assert single_layer_features.shape[0] == len(tokenized_sent)

            single_layer_features = torch.tensor(
                [np.mean(single_layer_features[untok_tok_mapping[i][0]:untok_tok_mapping[i][-1] + 1, :], axis=0)
                 for i in range(len(untokenized_sent))])

            assert single_layer_features.shape[0] == len(sent.split(' '))

            translation[layer_idx, :] -= single_layer_features.numpy().mean(axis=0)

    translation /= len(sentences)

    np.savez(args.translation_npz, translation)
