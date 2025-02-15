import argparse
import os
import sys
import math
from itertools import cycle
from typing import Union

sys.path.append(os.getcwd())


import h5py
import cv2
import torch
import numpy as np
import openslide

from preprocess.create_coords import knn, fps


data_root = '/data/smb/syh/WSI_cls/'

def patch_coords(h5_file):
    with h5py.File(h5_file, 'r') as f:
            return torch.tensor(f['coords'])

def hex_to_bgr(hex_color):
    # 去掉十六进制颜色前缀‘#’
    hex_color = hex_color.lstrip('#')
    # 解析十六进制颜色值，并转换为整数
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # 调整颜色顺序从RGB到BGR
    bgr = (rgb[2], rgb[1], rgb[0])
    return bgr

def draw_points(img_path, coords, k):
    colors = [
        '#FFFF08',
        '#00FF00',
        '#08FFFF',
        '#FF00FF',
        '#FF0000',
        '#0808FF',
    ]

    wsi = openslide.OpenSlide(img_path)

    level = 0
    for lvl, dims in enumerate(wsi.level_dimensions):
        level = lvl
        if max(dims) < 8000:
            break

    # print(len(wsi.level_dimensions))
    dims = wsi.level_dimensions[level]
    downsample_factor = wsi.level_downsamples[level]

    img_org = wsi.read_region((0, 0), level, dims).convert('RGB')
    img_org = np.array(img_org, dtype=np.uint8)
    img_org = img_org[:, :, ::-1]

    img = img_org

    overlay = np.zeros(img.shape, dtype=np.uint8)
    num_instances = coords.shape[0]
    for idx, color in zip(range(0, num_instances, k), cycle(colors)):
        sub_coord = coords[idx : (idx + k)]

        for coord in sub_coord:
            coord = coord / downsample_factor
            coord = coord.tolist()
            coord = [int(c) for c in coord]
            coord2 = [int(c + 512 / downsample_factor) for c in coord]
            # print(coord, hex_to_bgr(color))
            # cv2.circle(img, coord, 3, hex_to_bgr(color), 3)
            cv2.rectangle(overlay, coord, coord2, hex_to_bgr(color), -1)

    # 将overlay层与原图合并
    alpha = 0.8
    img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)
    return img, img_org


# def fps(points, n_samples):
#     """
#     points: [N, 3] array containing the whole point cloud
#     n_samples: samples you want in the sampled point cloud typically << N
#     """
#     points = np.array(points, dtype=np.int64)

#     # Represent the points by their indices in points
#     points_left = np.arange(len(points)) # [P]

#     # Initialise an array for the sampled indices
#     sample_inds = np.zeros(n_samples, dtype='int') # [S]

#     # Initialise distances to inf
#     dists = np.ones_like(points_left) * float('inf') # [P]

#     # Select a point from points by its index, save it
#     selected = 0
#     sample_inds[0] = points_left[selected]

#     # Delete selected
#     points_left = np.delete(points_left, selected) # [P - 1]

#     # Iteratively select points for a maximum of n_samples
#     for i in range(1, n_samples):
#         # Find the distance to the last added point in selected
#         # and all the others
#         last_added = sample_inds[i-1]

#         dist_to_last_added_point = ((points[last_added] - points[points_left]) ** 2).sum(-1)

#         # If closer, updated distances
#         dists[points_left] = np.minimum(dist_to_last_added_point,
#                                         dists[points_left]) # [P - i]

#         # We want to pick the one that has the largest nearest neighbour
#         # distance to the sampled points
#         selected = np.argmax(dists[points_left])
#         sample_inds[i] = points_left[selected]

#         # Update points_left
#         points_left = np.delete(points_left, selected)

#     return sample_inds

# def knn(coords, idx, k):
#     """
#         return the index of each instances.
#     """

#     output = []
#     coords = coords.long()
#     for i in idx:
#         diff = []
#         centroid = coords[i]
#         diff = ((centroid - coords) ** 2).sum(-1)
#         length = diff.shape[0]
#         indeces = torch.tensor(diff).topk(k=min(k, length), largest=False)[1]
#         indeces = indeces[1:]  # the nearest point is itself, remove it.

#         output.append(torch.tensor([i]))
#         output.append(indeces)

#     output = torch.cat(output)

#     remaining = []
#     for i in range(len(coords)):
#         if (i == output).sum() == 0:
#             remaining.append(i)

#     print(remaining, len(coords))
#     output = torch.cat([output, torch.tensor(remaining)])

#     return output

def showArrayOrTensor(t:Union[np.ndarray|torch.Tensor],tag=""):
    if isinstance(t,torch.Tensor):
        print(f"tag:{tag}, size:{t.shape}, type:{t.dtype}, device:{t.device}")
    elif isinstance(t,np.ndarray):
        print(f"tag:{tag}, size:{t.shape}, type:{t.dtype}")
    else:
        print(f"Tag({tag}) is {type(t)}, it's not a array or tensor!")
    
def main():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_root", default='/data/smb/syh/WSI_cls/')
    parser.add_argument('--dataset', default='cam16', type=str)
    parser.add_argument('--k', default=64, type=int)
    args = parser.parse_args()

    assert args.dataset in ['cam16', 'brac', 'lung']

    h5_dir = os.path.join(data_root, args.dataset, 'feat_clam_resnet50', 'patch_size_256_at_mag_20', 'h5_files')
    img_dir = os.path.join(data_root, args.dataset, 'img')

    os.makedirs('tmp', exist_ok=True)

    for slide in os.listdir(img_dir):

        slide_id, _ = os.path.splitext(slide)

        h5_path = os.path.join(h5_dir, slide_id + '.h5')
        coords = patch_coords(h5_path)

        idx = fps(coords, math.ceil(len(coords) / args.k))
        idx = knn(coords, idx, k=args.k)

        coords = coords[idx]

        img_path = os.path.join(img_dir, slide)
        img, img_org = draw_points(img_path, coords, args.k)

        showArrayOrTensor(img,'img') 
        showArrayOrTensor(idx,'idx')
        showArrayOrTensor(coords,'coords')
        showArrayOrTensor(slide,'slide')
        print( h5_path)
        
        input("pause")


        cv2.imwrite('tmp/{}.jpg'.format(slide_id), img)
        cv2.imwrite('tmp/{}_org.jpg'.format(slide_id), img_org)




main()