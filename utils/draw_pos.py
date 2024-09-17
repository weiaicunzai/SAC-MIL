import argparse
import glob
import os

import torch
import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_pos(data_path):
    data = []
    # print(os.path.join(data_path, '**', '.*.pt'))
    for i in glob.iglob(os.path.join(data_path, 'pos', '*.pt'), recursive=True):
        # print(i)
        pos = torch.load(i)
        data.append(pos)

        # if count > 100:
            # break

    return torch.cat(data, dim=0).numpy()

def load_data(data_path):
    res = []
    count_x = 0
    for i in glob.iglob(os.path.join(data_path, 'feat_clam_resnet50', 'patch_size_256_at_mag_20', 'h5_files', '*.h5'), recursive=True):
        with h5py.File(i, 'r') as f:
            # print(f['coords'].shape)
            data = np.array(f['coords'], dtype=np.float32)
            # print((data[0] == 0).sum())
            # print((data[1] == 0).sum())
            # count_x += (data[:, 0] == 0).sum()
            # count_y += (data[:, 1] == 0).sum()
            # data

            data[:, 0] = data[:, 0] / 256
            data[:, 1] = data[:, 1] / 256
            mask = np.all(data == [0, 0], axis=1)
            count_x += mask.sum()


            res.append(data)


    # print(count_x, count_y)
    print(count_x)


    res = np.concatenate(res, axis=0)

    return res


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_root", default='/data/smb/syh/WSI_cls/')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--norm', action='store_true', help='plot nomralized fig')
    args = parser.parse_args()

    assert args.dataset in ['cam16', 'lung']

    data_path = os.path.join(args.data_root, args.dataset)
    if args.norm:
        data = load_pos(data_path)
    else:
        data = load_data(data_path)

    plt.hist2d(data[:, 0], data[:, 1], bins=100, cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('Counts in bin')
    if args.norm:
        plt.savefig('{}_norm.jpg'.format(args.dataset))
    else:
        plt.savefig('{}_idx.jpg'.format(args.dataset))



if __name__ == '__main__':
    main()
