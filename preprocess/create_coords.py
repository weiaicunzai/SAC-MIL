import argparse
import os
import glob
import math
import random

import h5py
import torch
import numpy as np


data_root = '/data/smb/syh/WSI_cls/'

def patch_coords(h5_file):
    with h5py.File(h5_file, 'r') as f:
            # 读取数据集
            # dataset = f.keys()
            # print(dataset)
            # print(f['coords'])

            return torch.tensor(f['coords'])


    # data = patch_coords(data_root)
def norm_coords(coords):
    #  max
    # pass
    # print(type(coords))
    # print(coords.shape)
    # num_coords = len(coords)
    # for i in coords:
        # print(coords[i])
        # print(i.shape)

    max_x1 = coords[:, 0].max()
    max_y1 = coords[:, 1].max()
    min_x1 = coords[:, 0].min()
    min_y1 = coords[:, 1].min()

    min_coords = torch.tensor([min_x1, min_y1], dtype=torch.float)
    max_coords = torch.tensor([max_x1, max_y1], dtype=torch.float)
    coords = (coords - min_coords) / (max_coords - min_coords)
    return coords

def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points, dtype=np.int64)
    # points = points.astype(np.int64)

    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    # selected = np.random.choice(points_left)
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]

        dist_to_last_added_point = ((points[last_added] - points[points_left]) ** 2).sum(-1)

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return sample_inds


def assert_indices(masks, indices):
    for idx in indices:
        assert (idx == masks).sum() == 0
        # if idx not in masks:
            # print(idx)


def knn_no_overlap(coords, idx, k):

    output = []
    coords = coords.long()
    mask = torch.tensor(idx).long() # used instances
    remain_instances = coords.shape[0]
    pdist = nn.PairwiseDistance(p=2)


    for i in idx:
        diff = []
        centroid = coords[i].clone()

        tmp_coords = coords[mask].clone()
        coords[mask] = centroid + 300000
        coords[i] = centroid

        # print(coords[i], centroid, mask, i, (mask == i).sum(), idx)
        # continue
        # coords[i] = centroid
        # print(i, output)
        # print('centroid', centroid)
        # print('mask', mask)
        # print(coords)
        # print('ffff')
        # print(tmp_coords)
        # print('-----')
        # print(coords)
        # print('coords', i, coords)
        # print(centroid)

        # diff1 = ((centroid - coords) ** 2).sum(-1)
        diff = pdist(centroid, coords)
        # print((diff - diff1).mean())
        # print((diff - diff1).mean())
        # import sys; sys.exit()
        assert remain_instances >= 0
        if remain_instances > k:
            tmp_k = k
        else:
            tmp_k = remain_instances
        # indeces = torch.tensor(diff).topk(k=k, largest=False)[1]
        # print(tmp_k)
        indeces = torch.tensor(diff).topk(k=tmp_k, largest=False)[1]
        # print(centroid)
        # print(coords)
        # print(diff)
        # print(indeces[0], i, '-----')
        # print(indeces)
        # print(coords[indeces[0]], coords[i], centroid)
        # print((mask == indeces[0]).sum())
        assert indeces[0] == i
        indeces = indeces[1:]  # the nearest point is itself, remove it.


        output.append(torch.tensor([i]))
        output.append(indeces)

        # coords[mask] = tmp_coords
        # print(mask)

        # print(indeces, i)
        # print((mask[indeces] == 1).long())
        # assert mask[indeces].sum() == 0
        assert_indices(mask, indeces)
        coords[mask] = tmp_coords
        # mask[indeces] = 1
        # mask[i] = 1
        remain_instances -= k
        # print(remain_instances)

        # mask[torch.cat(output)] = 1
        # print(mask., indeces.dtype)
        mask = torch.cat([mask, indeces])

    output = torch.cat(output)

    # import sys; sys.exit()
    # remaining = []
    # for i in range(len(coords)):
    #     if (i == output).sum() == 0:
    #         remaining.append(i)
    # output = torch.cat([output, torch.tensor(remaining)])

    return output

def knn(coords, idx, k):
    # ddd
    # centroid = coords[idx]
    # coords = np.array(coords, dtype=np.uint64)
    output = []
    coords = coords.long()
    # print(idx)
    # print(len(coords), 'ffff')
    for i in idx:
        # print(i)
        diff = []
        # print(coords.shape, i)
        centroid = coords[i]
        # for coord in coords:
        #     diff.append(
        #         np.linalg.norm(centroid - coord)
        #     )

        diff = ((centroid - coords) ** 2).sum(-1)
        indeces = torch.tensor(diff).topk(k=min(k, diff.shape[0]), largest=False)[1]
        indeces = indeces[1:]  # the nearest point is itself, remove it.

        output.append(torch.tensor([i]))
        output.append(indeces)



    output = torch.cat(output)

    remaining = []
    for i in range(len(coords)):
        if (i == output).sum() == 0:
            remaining.append(i)
    output = torch.cat([output, torch.tensor(remaining, dtype=torch.long)])

    return output

def gen_color():
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

# def draw_points(img_dir, h5_path, coords, color):
#     basename = os.path.basename(h5_path)
#     basename = basename.replace('h5', 'tif')

#     img_path = os.path.join(img_dir, basename)

#     wsi = openslide.OpenSlide(img_path)

#     level = 5
#     dims = wsi.level_dimensions[level]
#     downsample_factor = wsi.level_downsamples[level]

#     # dims = dims / downsample_factor
#     img = wsi.read_region((0, 0), level, dims).convert('RGB')
#     img = np.array(img)
#     img = img[:, :, ::-1]

#     # print(type(img), img[33, 3])
#     # print(img.shape, img.dtype)
#     for coord in coords:
#         coord = coord / downsample_factor
#         # print(type(coord))
#         coord = coord.tolist()
#         coord = [int(c) for c in coord]
#         # coord = [30, 30]
#         # img = np.zeros((400, 400, 3), np.uint8)
#         cv2.circle(img, coord, 3, color, 3)

#     # print(img.max())
#     # print(img.shape, img.dtype)
#     return img

def main():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_root", default='/data/smb/syh/WSI_cls/')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--k', required=True, type=int)
    args = parser.parse_args()

    assert args.dataset in ['cam16', 'lung', 'brac']

    h5_dir = os.path.join(data_root, args.dataset, 'feat_clam_resnet50', 'patch_size_256_at_mag_20', 'h5_files')

    idx_path = os.path.join(data_root, args.dataset, 'idx_{}'.format(args.k))
    pos_path = os.path.join(data_root, args.dataset, 'pos')
    os.makedirs(idx_path, exist_ok=True)
    os.makedirs(pos_path, exist_ok=True)
    for h5_path in glob.iglob(os.path.join(h5_dir, '**', '*.h5'), recursive=True):
        coords = patch_coords(h5_path)

        # coords = norm_coords(coords)
        idx = fps(coords, math.ceil(len(coords) / args.k))
        idx = knn(coords, idx, k=args.k)

        idx_file = os.path.join(idx_path, os.path.basename(h5_path).replace('.h5', '.pt'))
        torch.save(idx, idx_file)
        print(idx_file)

        # process and save coords
        coords = norm_coords(coords)
        pos_file = os.path.join(pos_path, os.path.basename(h5_path).replace('.h5', '.pt'))
        torch.save(coords, pos_file)


if __name__ == '__main__':
    main()