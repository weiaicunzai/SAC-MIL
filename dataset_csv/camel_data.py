import random
import time
import torch
import pandas as pd
from pathlib import Path
import functools
import pickle
import os
import sys
from collections import defaultdict
import multiprocessing as mp
import copy
sys.path.append(os.getcwd())

import numpy as np
import torch.utils.data as data
# from torch.utils.data import dataloader
# from torch.utils.data import IterableDataset, default_collate
import pandas as pd
import math
import lmdb
from struct import unpack
from torch.utils.data import default_collate

from datasets.wsi import WSIJSON, WSIJSONLazyInitializer



def get_wsi(slide, json_dir, lazy):
    json_path = os.path.join(json_dir, slide + '.json')
    if not lazy:
        return WSIJSON(json_path=json_path, direction=0)
    else:
        return WSIJSONLazyInitializer(json_path=json_path, direction=0)

class WSIDataset33(data.IterableDataset):

    def __init__(self, settings, data_set, fold, batch_size, drop_last=False, allow_reapt=False, dist=None, lazy=True, replica=2):
        """the num_worker of each CAMLON16 dataset is one, """
        # assert data_set in ['train', 'val']
        print(self)

        assert data_set in ['train', 'val', 'test']


        self.batch_size = batch_size
        self.drop_last = drop_last
        self.allow_reapt = allow_reapt
        self.data_set = data_set
        self.fold = fold
        # self.trans = transforms
        # from conf.camlon16 import settings
        self.settings = settings
        self.lazy = lazy

        # self.orig_wsis = self.get_wsis(data_set=data_set)
        self.orig_wsis = self.get_wsis(data_set=self.data_set, fold=self.fold)
        print('total number of wsis: {}'.format(len(self.orig_wsis)))
        self.env = lmdb.open(settings.feat_dir, readonly=True, lock=False)
        self.seed = 52

        self.dist = dist

        self.max_len = settings.max_len
        self.seq_len = 1024
        num_chunks = math.ceil(self.max_len / self.seq_len)
        # self.wsi_len = 79872
        self.wsi_len = num_chunks * self.seq_len
        # self.wsi_len = num_chunks * self.seq_len
        print('wsi length: {}'.format(self.wsi_len))

        #print('loading all the data')
        #self.cache = self.load_data()
        #print('done')
        self.cache = {}

        self.if_shuffle = True

        self.replica = replica


    def shuffle_coords(self, wsi):
        # wsi = copy.deepcopy(wsi)

        # for i in range(len(wsi.coords)):
        #     random.shuffle(wsi.coords[i])
        random.shuffle(wsi.coords[wsi.direction])

        return wsi

    def load_data(self):
        # with self.env.begin(write=False):
        #     cursor = txn.cursor()
        #     for key, _ in cursor:
        #         keys.append(key)

        res = {}
        for idx, wsi in enumerate(self.orig_wsis):
            for data in wsi:
                patch_id = data['patch_id']
                feat = self.cache.get(patch_id, None)
                if feat is None:
                    res[patch_id] = self.read_feat(patch_id)
                else:
                    res[patch_id] = feat

            # print(idx)
        #res = {}
        #for key in keys:
        return res

    def split_wsis(self, wsis):
        if not self.dist.is_initialized():
            return wsis
        else:
            """get wsis for each gpu"""
            rank = self.dist.get_rank()
            num_replicas = self.dist.get_world_size()
            num_samples = math.ceil(len(wsis) / num_replicas)
            subsample = wsis[rank * num_samples: (rank + 1) * num_samples]

            # make sure each gpu acclocated the same number of wsis
            if len(subsample) < num_samples:
                diff = num_samples - len(subsample)
                subsample.extend(wsis[:diff])

            return subsample


    def get_wsis(self, data_set, fold):
        # wsis = self.data_set()
        # from conf.camlon16 import settings
        file_list = pd.read_csv(self.settings.file_list_csv)
        file_list['slide_id'] = file_list['slide_id'].apply(
            lambda x: os.path.splitext(x)[0]
        )
        # print(file_list['slide_id'])
        # splits = os.path.join(os.pat)
        # split_file = os.path.join('datasets', 'splits', 'cam16', 'splits_{}.csv'.format(fold))
        # pd.set_option('display.max_rows', None)
        split_file = os.path.join(self.settings.split_dir, 'splits_{}.csv'.format(fold))
        splits = pd.read_csv(split_file)
        # data_list = splits[data_set]
        # if data_set != 'test':
        train_split = splits['train'].dropna()
        val_split = splits['val'].dropna()
        test_split = splits['test'].dropna()
        # print(data_list)
        # print(val_split, train_split, test_split)


        # if data_set != 'test':
        if data_set == 'train':
            mask1 = file_list['slide_id'].isin(train_split)
            mask2 = file_list['slide_id'].isin(val_split)
            slide_ids = file_list['slide_id'][mask1 | mask2]
        else:
            mask1 = file_list['slide_id'].isin(test_split)
            slide_ids = file_list['slide_id'][mask1]


        # print(slide_ids)
        wsis = []



        # pool = mp.Pool(processes=mp.cpu_count())
        print('start load json')
        t1 = time.time()
        # pool = mp.Pool(processes=mp.cpu_count())
        # pool = mp.Pool(processes=4)
        # fn = functools.partial(get_wsi, json_dir=self.settings.json_dir)
        # wsis = pool.map(fn, slide_ids.tolist())
        wsis = []
        # wsis = {}
        for slide_id in slide_ids:
            wsi = get_wsi(slide_id, self.settings.json_dir, self.lazy)
            wsis.append(wsi)

            # if data_set == 'train':
            #     if len(wsis) == 10:
            #         break

            # else:
            #     if len(wsis) == 3:
            #         break

            # if len(wsis) == 3:
            #     break

        # wsis = wsis[:5]
        print('done load json {}'.format(time.time() - t1))

        return wsis




    def set_direction(self, wsis, direction):
        for wsi in wsis:
            wsi.direction = direction

        return wsis

    def set_random_direction(self, wsis):
        random.seed(self.seed)
        for wsi in wsis:
            direction = random.randint(0, 7)
            wsi.direction = direction
        return wsis

    def orgnize_wsis(self, orig_wsis):
        wsis = []
        # when batch size is larger than the total num of wsis
        if self.batch_size > len(orig_wsis):
            if self.allow_reapt:
                wsis = orig_wsis
                while len(wsis) != self.batch_size:
                    wsis.extend(random.sample(orig_wsis, k=1))
            else:
                raise ValueError('allow_reapt should be True when batch_size is larger than the whole wsis')

        else:
            remainder = len(orig_wsis) % self.batch_size
            # if the total number of wsis is not divisible by batch_size
            if remainder > 0:
                # if we do not drop the last, we randomly select "self.batch_size - remainder" number of
                # samples add to the orig_
                random.seed(self.seed)
                if not self.drop_last:
                    wsis = orig_wsis
                    # for wsi in random.sample(self.orig_wsis, k=self.batch_size - remainder):
                    for wsi in random.sample(orig_wsis, k=self.batch_size - remainder):
                        wsis.append(wsi)

                else:
                    # if drop last, we randomly sample "total number of self.orig_wsis - remainer" number
                    # of wsis
                    wsis = random.sample(orig_wsis, k=len(orig_wsis) - remainder)
                    assert len(wsis) == len(orig_wsis) - remainder
            else:
                wsis = orig_wsis
        return wsis


    def shuffle(self, wsis):
        """manually shuffle all the wsis, because when num_workers > 0,
        the copy of dataset wont update in the main process """
        random.seed(self.seed)
        random.shuffle(wsis)

        return wsis

    def cal_seq_len(self, wsis):
        outputs = []

        for idx in range(0, len(wsis), self.batch_size):

            batch_wsi = wsis[idx : idx + self.batch_size]
            max_len = max([wsi.num_patches for wsi in batch_wsi])

            outputs.append(max_len)

        return outputs

    def cycle(self, iterable):
        while True:
            for data in iterable:
                yield data

    def read_feat(self, patch_id):
        with self.env.begin(write=False) as txn:
            img_stream = txn.get(patch_id.encode())
            feat = unpack('1024f', img_stream)
            feat = torch.tensor(feat)

        return feat

    def read_replica(self, wsis):

        output = {}
        output['feat'] = []
        output['p_label'] = []
        for wsi in wsis:

            data = self.read_seq(wsi)
            output['feat'].append(data['feat'])
            output['filename'] = data['filename']
            output['label'] = data['label']
            output['p_label'].append(data['p_label'])


        output['feat'] = torch.stack(output['feat'], dim=0)
        # print(output['feat'].shape)
        # import sys; sys.exit()
        output['p_label'] = torch.stack(output['p_label'])

        return output

    def read_seq(self, wsi):

        feats = []
        label = 0
        filename = ''

        p_labels = []
        for idx, d in enumerate(wsi):
            patch_id = d['patch_id']

            feat = self.cache.get(patch_id, None)

            if feat is None:
               feat = self.read_feat(patch_id)
               self.cache[patch_id] = feat

            p_labels.append(d['p_label'])

            feats.append(feat)
            label = d['label']
            filename = d['filename']

            if idx == self.seq_len - 1:
                break

        output = {}
        output['feat'] = torch.stack(feats, dim=0)
        output['label'] = int(label)
        output['filename'] = filename
        output['p_label'] = torch.tensor(p_labels)
        return output

    def shift_wsi(self, wsi, num_patch):

        shift = random.randint(0, num_patch - 1)
        for i in range(shift):
            next(wsi)

        return wsi

    def copy_wsi(self, wsi):
        shallow_wsi = copy.copy(wsi)
        shallow_wsi.coords[shallow_wsi.direction] = copy.deepcopy(wsi.coords[wsi.direction])

        return shallow_wsi


    def data_aug(self, batch_wsi):
        wsis = []
        num_patches = [w.num_patches for w in batch_wsi]
        random.seed(self.seed)

        for wsi, num_patch in zip(batch_wsi, num_patches):


            replicas = []

            for _ in range(self.replica):

                wsi_cp = wsi
                direction = random.randint(0, 7)
                wsi_cp.direction = direction

                # wsi_cp = copy.deepcopy(wsi_cp)
                wsi_cp = self.copy_wsi(wsi_cp)

                # shuffle
                wsi_cp = self.shuffle_coords(wsi_cp)

                # cycle
                wsi_cp = self.cycle(wsi_cp) # no problem

                # shift
                if self.data_set == 'train':
                    wsi_cp = self.shift_wsi(wsi_cp, num_patch)

                replicas.append(wsi_cp)

            wsis.append(replicas)

        return wsis

    def cache_wsis(self, wsis):
        for wsi in wsis:
            # json_path = wsi.json_path
            # basename = os.path.join(json_path)
            # basename = os.path.splitext(basename)[0] + '.'
            for d in wsi.data:
                patch_id = d['patch_id']
                with self.env.begin(write=False) as txn:
                    img_stream = txn.get(patch_id.encode())
                    feat = unpack('1024f', img_stream)
                    # feat = torch.tensor(feat)
                    self.cache[patch_id] = feat


    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        # create a new list to avoid change the self.orig_wsis
        # during each epoch
        wsis = []
        for wsi in self.orig_wsis:
           wsis.append(wsi)

        wsis = self.shuffle(wsis)


        # wsis = self.split_wsis(wsis) # used for ddp training
        # wsis = self.orgnize_wsis(wsis)
        # global_seq_len = self.cal_seq_len(wsis)

        if self.data_set != 'train':
            wsis = self.set_direction(wsis, direction=0)
            if self.drop_last:
                raise ValueError('during inference, the drop_last should not be set to true')
        else:
            wsis = self.set_random_direction(wsis)

        count = 0

        for idx in range(0, len(wsis), self.batch_size):#0
            # use cache to store the current wsi feats
            # since we are using a cyclic training procedure
            # features will be read multiple times from disk
            # therefore, cache current wsis will save some
            # I/O time.
            # The whole dataset is quit large 30GB - 100GB
            # storing all the dataset in the RAM would require
            # much more memory
            # clear cache

            self.cache = {}

            # add seeds here to avoid different seed value for
            # different workers if we set seed += 1024 at the
            # end of each data = next(x) loop (count might not
            # be divided by each )
            self.seed += 1024

            batch_wsi = wsis[idx : idx + self.batch_size]

            # self.cache_wsis(batch_wsi)


            if self.lazy:
                batch_wsi = [x.data() for x in batch_wsi]

            # batch_wsi = [self.shuffle_coords(x) for x in batch_wsi]


            # assert len(batch_wsi) == self.batch_size
            if len(batch_wsi) < self.batch_size:
                if self.drop_last:
                    continue

            # num_patches = [w.num_patches for w in batch_wsi]
            # batch_wsi = [self.cycle(x) for x in batch_wsi]

            # if self.data_set == 'train':
                # batch_wsi = self.shift_wsi(batch_wsi, num_patches)

            # max_len_idx = idx // self.batch_size
            # mex_len = self.wsi_len

            # if not global_seq_len[max_len_idx]:
                # warnings.warn('max batch len equals 0')
                # continue

            # max_len = global_seq_len[max_len_idx]

            # if wsi len is not divisible by num_workers,
            # the last few elements will
            # change the order of reading next round
            # set a global counter to eliminate this issue

            # print('before', len(batch_wsi))
            # print(t4)
            batch_wsi = self.data_aug(batch_wsi)
            # print('after', len(batch_wsi))
            # t5 = time.time()
            # print(t5 - t4, 'ssssss, all data_aug')
            for patch_idx in range(0, self.wsi_len, self.seq_len):#104

                    outputs = []
                    for wsi in batch_wsi:
                        # print(wsi)

                        # data = next(wsi)
                        data = self.read_replica(wsi)

                        if worker_info is not None:
                            if count % worker_info.num_workers != worker_info.id:
                                continue

                            #data['worker_id'] = worker_info.id
                            #data['count'] = self.count
                            #data['patch_idx'] = patch_idx
                            #data['seed'] = tmp_seed
                            # data['dir'] = tmp_dircts
                        # data = self.read_img(data)

                        # if patch_idx < max_len - 1:
                        # if patch_idx < self.wsi_len - 1:
                        if patch_idx < self.wsi_len - self.seq_len:
                            data['is_last'] = 0
                        else:
                            data['is_last'] = 1

                        outputs.append(data)

                    count += 1

                    if outputs:
                        # print(type(outputs), len(outputs), outputs[0]['feat'].shape)
                        # print(type(outputs[0]['label'])) # int
                        # print(type(outputs[0]['filename'])) # str
                        # print(type(outputs[0]['is_last']))

                        data = default_collate(outputs)
                        # print(data['feat'].shape)
                        # import sys; sys.exit()
                        # print(data['label'])
                        # print(data['feat'].shape)
                        # print(data['feat'][:, :, 0, 0])
                        # print(len(self.cache.keys()), 11111, 'ffffffffff')
                        # print(data['feat'].shape)
                        # import sys; sys.exit()
                        yield data['feat'], data['label'], data['filename'], data['is_last']
                        # , data['p_label']


class WSIDataset(data.IterableDataset):

    def __init__(self, settings, data_set, fold, batch_size, drop_last=False, allow_reapt=False, dist=None, lazy=True, replica=2):
        """the num_worker of each CAMLON16 dataset is one, """
        # assert data_set in ['train', 'val']
        print(self)

        assert data_set in ['train', 'val', 'test']


        self.batch_size = batch_size
        self.drop_last = drop_last
        self.allow_reapt = allow_reapt
        self.data_set = data_set
        self.fold = fold
        self.settings = settings
        self.lazy = lazy

        self.orig_wsis = self.get_wsis(data_set=self.data_set, fold=self.fold)
        print('total number of wsis: {}'.format(len(self.orig_wsis)))
        self.seed = 52

        self.dist = dist

        self.max_len = settings.max_len
        self.seq_len = 1024
        num_chunks = math.ceil(self.max_len / self.seq_len)
        self.wsi_len = num_chunks * 2 * self.seq_len
        print('wsi length: {}'.format(self.wsi_len))

        self.cache = {}

        self.if_shuffle = True

        self.replica = replica

        self.label = {}

    def get_wsis(self, data_set, fold):
        file_list = pd.read_csv(self.settings.file_list_csv)
        file_list['slide_id'] = file_list['slide_id'].apply(
            lambda x: os.path.splitext(x)[0]
        )

        split_file = os.path.join(self.settings.split_dir, 'splits_{}.csv'.format(fold))
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

    def read_replica(self, feat):

        # copy feat to replica
        feats = feat.unsqueeze(dim=0)
        feats = feats.expand(self.replica, feats.shape[1], feats.shape[2])

        # generate indices using argsort
        indices = torch.argsort(torch.rand(*feats.shape[:2]), dim=1)
        indices = indices.unsqueeze(dim=-1).expand(*feats.shape)

        # extract feats using torch.gather
        feats = torch.gather(feats, dim=1, index=indices)

        return feats


    def pad_seq(self, features):
        fact = self.wsi_len / features.shape[1] + 1

        features = features.repeat(1, int(fact), 1)

        features = features[:, :self.wsi_len, :]

        assert features.shape[1] == self.wsi_len

        return features

    # def shuffle_tensor(self, index, feat):
    #     random.shuffle(index)
    #     test = feat[index]

    #     return test

    def read_data(self, slide_id):
        feat_path = self.settings.feat_dir
        feat = torch.load(os.path.join(feat_path, slide_id + '.pt'))

        return feat

    def __iter__(self):

        slide_names = list(self.orig_wsis.keys())

        if self.data_set == 'train':
            random.shuffle(slide_names)

        for slide_id in slide_names:

            output = self.read_data(slide_id)
            label = self.orig_wsis[slide_id]

            features = self.read_replica(output)

            # features = self.pad_seq(features)

            num_chunks = self.wsi_len / self.seq_len
            count = 0
            for chunk in features.chunk(int(num_chunks), dim=1):
                count += 1
                if count == num_chunks:
                    is_last = 1
                else:
                    is_last = 0

                data = default_collate([{'feat': chunk, 'label': label, 'filename': slide_id, 'is_last': is_last}])

                yield data['feat'], data['label'], data['filename'], data['is_last']
