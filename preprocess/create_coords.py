import argparse
import os
import glob
import math

import h5py
import cv2
import torch
import numpy as np
import openslide


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

    min_coords = torch.tensor([min_x1, min_y1])
    max_coords = torch.tensor([max_x1, max_y1])
    coords = (coords - min_coords) / max_coords
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
        # print(dist_to_last_added_point2)

        # print(dist_to_last_added_point, 'ccc')
        # print(dist_to_last_added_point ** 2, 'ccc')
        # print(points[last_added].shape, points[points_left].shape)
        # print(dist_to_last_added_point.shape)
        # print(dist_to_last_added_point * dist_to_last_added_point)
        # print(dist_to_last_added_point.dtype)
        # import sys; sys.exit()


        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    # return points[sample_inds]
    return sample_inds

def knn(coords, idx, k):
    # ddd
    # centroid = coords[idx]
    # coords = np.array(coords, dtype=np.uint64)
    output = []
    coords = coords.long()
    for i in idx:
        # print(i)
        diff = []
        centroid = coords[i]
        # print(type(centroid))
        # for coord in coords:
        #     diff.append(
        #         np.linalg.norm(centroid - coord)
        #     )

        # print(centroid.shape)
        # print(coords.shape)
        # print((centroid - coords).shape)

        diff = ((centroid - coords) ** 2).sum(-1)
        # print(diff)
        indeces = torch.tensor(diff).topk(k=k, largest=False)[1]
        indeces = indeces[1:]  # the nearest point is itself, remove it.

        # print(diff[indeces[0]])
        output.append(torch.tensor([i]))
        output.append(indeces)

    output = torch.cat(output)

    remaining = []
    for i in range(len(coords)):
        if (i == output).sum() == 0:
            # print(i, output)
            remaining.append(i)
    output = torch.cat([output, torch.tensor(remaining)])

    return output


def draw_points(img_dir, h5_path, coords, color):
    basename = os.path.basename(h5_path)
    basename = basename.replace('h5', 'tif')

    img_path = os.path.join(img_dir, basename)

    wsi = openslide.OpenSlide(img_path)

    level = 5
    dims = wsi.level_dimensions[level]
    downsample_factor = wsi.level_downsamples[level]

    # dims = dims / downsample_factor
    img = wsi.read_region((0, 0), level, dims).convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1]

    # print(type(img), img[33, 3])
    # print(img.shape, img.dtype)
    cv2.imwrite('test.jpg', img)
    img = cv2.imread('test.jpg')
    for coord in coords:
        coord = coord / downsample_factor
        # print(type(coord))
        coord = coord.tolist()
        coord = [int(c) for c in coord]
        # coord = [30, 30]
        # img = np.zeros((400, 400, 3), np.uint8)
        cv2.circle(img, coord, 3, color, 3)

    # print(img.max())
    # print(img.shape, img.dtype)
    return img

def main():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_root", default='/data/smb/syh/WSI_cls/')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--k', required=True, type=int)
    args = parser.parse_args()

    h5_dir = os.path.join(data_root, args.dataset, 'feat_clam_resnet50', 'patch_size_256_at_mag_20', 'h5_files')

    img_dir = os.path.join(data_root, args.dataset, 'img')

    idx_path = os.path.join(data_root, args.dataset, 'idx_{}'.format(args.k))
    os.makedirs(idx_path, exist_ok=True)
    for h5_path in glob.iglob(os.path.join(h5_dir, '**', '*.h5'), recursive=True):
        coords = patch_coords(h5_path)

        # img = draw_points(img_dir, h5_path, coords, (0, 0, 255))

        # coords = norm_coords(coords)
        # len(coords) / args.k
        idx = fps(coords, math.ceil(len(coords) / args.k))
        idx = knn(coords, idx, k=args.k)

        # print(idx.shape)
        # img = draw_points(img_dir, h5_path, coords, (0, 0, 255))
        # cv2.imwrite('test.jpg', img)

        # import sys; sys.exit()

        # torch.save()
        # idx_file = os.path.join(idx_path, os.path.basename(h5_path).replace('.h5', '.pt'))
        # torch.save(idx, idx_file)
        # print(idx_file)

        coords = norm_coords(coords)
        # print(coords[idx])
        # coords = coords[idx]
        # print(idx.shape)
        # print(coords.shape)
        print(coords)




main()