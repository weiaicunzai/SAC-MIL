import os
import random
import torch
import pandas as pd

import torch.utils.data as data

class WSIDataset(data.Dataset):
    def __init__(self, data_root, state, fold, dataset):
        super().__init__()
        # Set all input args as attributes
        self.data_root = data_root

        data_set = state

        self.feat_dir = os.path.join(
            self.data_root,
                # 'feat_clam_resnet50',
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

        # dataset_name = os.path.basename(self.data_root)
        dataset_name = dataset
        self.file_list_csv = os.path.join('dataset_csv', 'dataset_list', dataset_name, '{}.csv'.format(dataset_name))
        self.split_dir = os.path.join('dataset_csv', 'splits',  dataset_name)
        wsis = self.get_wsis(data_set, fold)
        self.wsis = wsis
        self.slide_names = list(self.wsis.keys())

        self.data_set = data_set

        self.tmp_data = {}

        # for idx in range(len(self)):
        #     slide_id = self.slide_names[idx]
        #     # label = self.wsis[slide_id]
        #     full_path = os.path.join(self.feat_dir, '{}.pt'.format(slide_id))
        #     self.get_data(full_path)



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

    def dict2tensor(self, features):
        assert isinstance(features, dict)
        return torch.tensor(list(features.values()))

    def get_data(self, full_path):
        features = self.tmp_data.get(full_path, None)
        # print('start loading...')

        if features is None:
            features = torch.load(full_path)
            if isinstance(features, dict):
                features = self.dict2tensor(features)
            self.tmp_data[full_path] = features

        # print('done loading.......')
        return features

    def __len__(self):
        return len(self.wsis.keys())

    # def __getitem__(self, idx):
    #     slide_id = self.slide_names[idx]
    #     label = self.wsis[slide_id]
    #     full_path = os.path.join(self.feat_dir, '{}.pt'.format(slide_id))

    #     features = self.get_data(full_path)

    #     return features, 1, label

    def __getitem__(self, idx):
        slide_id = self.slide_names[idx]
        label = self.wsis[slide_id]
        full_path = os.path.join(self.feat_dir, '{}.pt'.format(slide_id))
        # print(full_path)

        # features = torch.load(full_path)
        features = self.get_data(full_path)


        idx = torch.load(os.path.join(self.idx, slide_id + '.pt')).long()
        pos = torch.load(os.path.join(self.pos, slide_id + '.pt'))
        # pos_len = pos.shape[0]
        # feat_len = features.shape[0]
        # if pos_len > feat_len:
        #     pos = pos[:feat_len]
        # else:
        #     repeat_times = feat_len // pos_len
        #     remainder = feat_len % pos_len

        #     # 重复 pos 并拼接
        #     pos = torch.cat([pos] * repeat_times + [pos[:remainder]], dim=0)

        features = features.clone()

        assert pos.shape[0] == features.shape[0]
        if self.data_set == 'train':
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]
            pos = pos[index]

        assert pos.shape[0] == features.shape[0]
        return features, pos, label