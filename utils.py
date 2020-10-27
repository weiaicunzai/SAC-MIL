import os
import glob

import numpy as np
import torch.nn as nn

import transforms
from conf import settings

def test_trans():
    #trans = transforms.Compose)
    trans = transforms.Compose([
        [
            transforms.ToTensor(),
            transforms.Normalize(settings.MEAN, settings.STD)
        ]
    ])

    return trans

def training_trans():
    trans = transforms.Compose([
        transforms.RandomApply(
            [transforms.RandomRotation(30)],
            0.2
        ),
        transforms.RandomApply(
            [transforms.RandomHorizontalFlip()],
            0.2
        ),
        transforms.RandomApply(
            [transforms.ColorJitter()],
            0.15
        ),
        transforms.RandomApply(
            [transforms.RandomScale()],
            0.25
        ),
        transforms.RandomChoice([
            transforms.RandomApply(
                [transforms.GaussianNoise()],
                0.15
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur()],
                0.2
            )
        ]),
        transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD)
    ])

    return trans

def istrained(fold_idx):
    weights_dir = settings.CHECKPOINT_FOLDER
    if os.path.exists(weights_dir):
        return True

    glob_path = ''
    # find finished training weights

    for path in glob.iglob(os.path.join(settings.CHECKPOINT_FOLDER, '**', '*.pth'), recursive=True):
        print(path)

def get_model(args):
    if args.net == 'unet2d':
        from models.unet import UNet
        net = UNet(1)

    if args.gpu:
        net = net.cuda()

    if isinstance(args.gpu_ids, list):
        net = nn.DataParallel(net, device_ids=args.gpus)

    return net

def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    Args:
        dataset: instance of torch.nn.Dataset
    Returns:
        return: mean and std of this dataset
    """

    mean = 0
    std = 0

    for img, _ in dataset:
        mean += np.mean(img, axis=(0, 1))

    mean /= len(dataset)

    diff = 0
    for img, _ in dataset:

        diff += np.sum(np.power(img - mean, 2), axis=(0, 1))

    N = len(dataset) * np.prod(img.shape[:2])
    std = np.sqrt(diff / N)

    mean = mean / 255
    std = std / 255

    return np.array(mean), np.array(std)






