import torch
from torch.utils.data import DataLoader

import pickle
import numpy as np
from . import dataset


def build_loader(config):
    dataset_train, dataset_test = build_dataset(config)
    dataloader_train = DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE, collate_fn=dataset_train.collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=config.DATA.BATCH_SIZE, collate_fn=dataset_test.collate_fn)
    return dataloader_train, dataloader_test


def build_dataset(config):
    with open(config.DATA.DATA_PATH,'rb') as f:
        DataMap = pickle.load(f)
    dataset_train = dataset.ERCDATASET(DataMap['train'], config.DATA.USE_BERT, config.DATA.AUDIO_MAX_LENGTH)
    dataset_test = dataset.ERCDATASET(DataMap['test'], config.DATA.USE_BERT, config.DATA.AUDIO_MAX_LENGTH) 
    return dataset_train, dataset_test



