
from argparse import ArgumentParser
import yaml
import numpy as np
import os

import data
import task


if __name__ == 'main':
    argp = ArgumentParser()
    argp.add_argument('yaml_file')
    args = argp.parse_args()

    yaml_args = yaml.load(open(args.yaml_file))
    dataset = data.BERTDataset(yaml_args, task.ParseDistanceTask)
    assert len(dataset.dev_dataset.observations) == len(dataset.test_dataset.observations), \
        "DEV and TEST sets need to be parallel and hav the same sentence number."

    translation = np.zeros((768,))

    for dev_obs, test_obs in zip(dataset.dev_dataset.observations, dataset.test_dataset.observations):
        dev_emb = dev_obs.embeddings.numpy()
        test_emb = test_obs.embeddings.numpy()
        dev_emb = dev_emb.mean(axis=0)
        test_emb = test_emb.mean(axis=0)
        translation = translation + dev_emb - test_emb

    translation /= len(dataset.dev_dataset.observations)
    translation_path = os.path.join(yaml_args['embeddings']['root'], yaml_args['embeddings']['translation'])
    np.savez(translation_path, translation)
