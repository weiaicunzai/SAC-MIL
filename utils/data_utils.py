import logging
import os
import random

import pandas as pd

import torch
import math
from conf import settings

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger(__name__)


# def cal_wsi_len(max_len, num_seg):
#     return math.ceil(max_len / num_seg)



class WSIDataset(Dataset):
    def __init__(self, data_root, data_set, fold, wsi_len):
        # Set all input args as attributes
        self.data_root = data_root
        self.feat_dir = os.path.join(
            self.data_root,
                'feat_clam_resnet50',
                'patch_size_256_at_mag_20',
                'pt_files'
            )

        dataset_name = os.path.basename(self.data_root)
        self.file_list_csv = os.path.join('dataset_csv', 'dataset_list', dataset_name, '{}.csv'.format(dataset_name))
        self.split_dir = os.path.join('dataset_csv', 'splits',  dataset_name)
        wsis = self.get_wsis(data_set, fold)
        self.wsis = wsis
        self.slide_names = list(self.wsis.keys())

        # if dataset_name == 'cam16':
        #     self.wsi_len = 44380
        self.wsi_len = wsi_len



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
        # label = int(self.label[idx])
        label = self.wsis[slide_id]
        full_path = os.path.join(self.feat_dir, '{}.pt'.format(slide_id))
        features = torch.load(full_path)

        #----> shuffle
        # if self.shuffle == True:
        index = [x for x in range(features.shape[0])]
        random.shuffle(index)
        features = features[index]

        features = self.pad_seq(features)



        return features, label

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((args.img_size, args.img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])

    # trainset = datasets.CIFAR10(root="./data",
    #                             train=True,
    #                             download=True,
    #                             transform=transform_train)
    # testset = datasets.CIFAR10(root="./data",
    #                             train=False,
    #                             download=True,
    #                             transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.dataset_name == 'cam16':
        wsi_len = settings.cam16_wsi_len
    trainset = WSIDataset(
        os.path.join(args.data_root, args.dataset_name),
        data_set='train',
        fold=args.fold,
        wsi_len=wsi_len
    )
    testset = WSIDataset(
        os.path.join(args.data_root, args.dataset_name),
        data_set='test',
        fold=args.fold
    )
    if args.local_rank == 0:
        torch.distributed.barrier()

    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    # test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                            #   sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                            #  sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader



# dataset = WSIDataset('/data/smb/syh/WSI_cls/cam16', data_set='train', fold=0)
# print(dataset)

# import time
# t1 = time.time()
# for seq, label in dataset:
#     # print(seq.shape, label)
#     # print(seq)
#     pass

# print((time.time() - t1) / len(dataset))