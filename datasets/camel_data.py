import os
import random
import torch
import pandas as pd
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader


class WSIDataset(data.Dataset):
    def __init__(self, data_root, state, fold):
        super().__init__()
        # Set all input args as attributes
        self.data_root = data_root

        data_set = state

        self.feat_dir = os.path.join(
            self.data_root,
                'feat_clam_resnet50',
                'patch_size_256_at_mag_20',
                'pt_files'
            )
        self.idx = os.path.join(
            self.data_root,
            "idx_64"
        )

        self.pos = os.path.join(
            self.data_root,
            'pos'
        )

        dataset_name = os.path.basename(self.data_root)
        self.file_list_csv = os.path.join('dataset_csv', 'dataset_list', dataset_name, '{}.csv'.format(dataset_name))
        self.split_dir = os.path.join('dataset_csv', 'splits',  dataset_name)
        wsis = self.get_wsis(data_set, fold)
        self.wsis = wsis
        self.slide_names = list(self.wsis.keys())


        self.data_set = data_set



    def get_wsis(self, data_set, fold):
        file_list = pd.read_csv(self.file_list_csv)
        file_list['slide_id'] = file_list['slide_id'].apply(
            lambda x: os.path.splitext(x)[0]
        )

        split_file = os.path.join(self.split_dir, 'splits_{}.csv'.format(fold))
        splits = pd.read_csv(split_file)
        train_split = splits['train'].dropna()
        val_split = splits['val'].dropna()
        test_split = splits['test'].dropna()

        if data_set == 'train':
            mask1 = file_list['slide_id'].isin(train_split)
            mask2 = file_list['slide_id'].isin(val_split)
            slide_ids = file_list['slide_id'][mask1 | mask2]
        else:
            mask1 = file_list['slide_id'].isin(test_split)
            slide_ids = file_list['slide_id'][mask1]


        # slide_ids = slide_ids[:3]
        wsis = {}
        for slide_id in slide_ids:
            mask = file_list['slide_id'] == slide_id
            label = int(file_list[mask].iloc[0]['label'])
            wsis[slide_id] = label

        return wsis

    def pad_seq(self, features):
        fact = self.wsi_len / features.shape[0] + 1

        features = features.repeat(int(fact), 1)

        features = features[:self.wsi_len, :]

        assert features.shape[0] == self.wsi_len

        return features

    def __len__(self):
        return len(self.wsis.keys())

    def __getitem__(self, idx):
        slide_id = self.slide_names[idx]
        # print(slide_id)
        label = self.wsis[slide_id]
        full_path = os.path.join(self.feat_dir, '{}.pt'.format(slide_id))
        features = torch.load(full_path)

        # idx = torch.load(os.path.join(self.idx, slide_id + '.pt')).long()
        # features = features[idx]

        pos = torch.load(os.path.join(self.pos, slide_id + '.pt'))
        # pos = pos[idx]
        #----> shuffle
        # if self.shuffle == True:

        # features = self.pad_seq(features)

        if self.data_set == 'train':
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]
            pos = pos[index]

        return features, pos, label