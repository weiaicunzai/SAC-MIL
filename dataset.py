import os
import glob
import random

import cv2
import SimpleITK as sitk
import numpy as np

import torch
from torch.utils.data import Dataset

def normalize(img):
    img = img.astype(float)
    #if img.max() != 0:
    img = np.maximum(img,0) / img.max() * 255.0
    #else:
        #img = np.maximum(img,0)
    img = np.uint8(img)

    return img

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)

    ct_scan = sitk.GetArrayFromImage(itkimage)

    #print(itkimage.GetOrigin())
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def write_images(data_folder, dest_folder):
    for path in glob.iglob(os.path.join(data_folder, "*.mhd")):
        file_name = os.path.basename(path)
        itkimage = sitk.ReadImage(path)
        slices = sitk.GetArrayFromImage(itkimage)

        for idx, s in enumerate(slices):


            image = normalize(s)
        #print(image.shape)

            image_name = os.path.splitext(file_name)[0] + '_slice_' + str(idx) + '.png'
            image_path = os.path.join(dest_folder, image_name)
            if 'segmentation' in image_path:
                print(np.unique(image))
            cv2.imwrite(image_path, image)

#ct_scan, origin, spacing = load_itk('/home/baiyu/Downloads/promise12/TrainingData_Part3/Case38.mhd')
#print(type(ct_scan))
#print(ct_scan.shape)
#print(ct_scan[0].shape)
#img = normalize(ct_scan[0])
#
#cv2.imwrite('test.png', img)
#
#print(origin)
#print(spacing)
class Promise12(Dataset):

    def __init__(self, list_path, transforms=None):

        self.images = []
        self.masks = []
        #self.count = []
        #self.path = []
        self.fold_idx = os.path.basename(list_path).split('.')[0]

        self.transforms = transforms

        print('loading training data: {}'.format(list_path))
        with open(list_path) as f:
            #self.images = f.readlines().split('\n')
            for line in f.readlines():
                image_path = line.strip()
                image_dir = os.path.dirname(image_path)
                image_name = os.path.basename(image_path)
                mask_name = image_name.replace('_slice', '_segmentation_slice')
                mask_name = mask_name.replace('.jpg', '.png')
                mask_path = os.path.join(image_dir, mask_name)

                #self.images.append(image_path)
                #self.masks.append(mask_path)
                image = cv2.imread(image_path, 0)
                mask = cv2.imread(mask_path, 0)
                self.images.append(image)
                self.masks.append(mask)

        print('Done!')

    def __len__(self):
        return len(self.images)

        #search_path = os.path.join(path, '**', '*_segmentation.mhd')
        #for mask_path in glob.iglob(search_path, recursive=True):

        #    print(mask_path)
        #    # read mask
        #    mask_file = sitk.ReadImage(mask_path)
        #    mask = sitk.GetArrayFromImage(mask_file)

        #    # read image
        #    mask_name = os.path.basename(mask_path)
        #    image_name = mask_name.replace('_segmentation.mhd', '.mhd')
        #    image_path = os.path.join(os.path.dirname(mask_path), image_name)
        #    image_file = sitk.ReadImage(image_path)
        #    image = sitk.GetArrayFromImage(image_file)

        #    #print(mask.shape, image.shape)
        #    assert image.shape == mask.shape
        #    self.images.append(self.normalize(image))
        #    self.masks.append(np.uint8(mask))
        #    self.count.append(image.shape[0])
        #    #print(type(mr_scan), mr_scan.shape)

    #def normalize(self, img):
    #    img = img.astype(float)
    #    #if img.max() != 0:
    #    img = np.maximum(img,0) / img.max() * 255.0
    #    #else:
    #        #img = np.maximum(img,0)
    #    img = np.uint8(img)

    #    return img

    #def _index(self, index):

    #    # convert index to number
    #    tmp_number = index + 1
    #    for study_idx, slice_num in enumerate(self.count):

    #        tmp_number -= slice_num

    #        # in current interval
    #        if tmp_number <= 0:

    #            # convert slice number back to index
    #            slice_idx =  tmp_number + slice_num - 1
    #            return study_idx, slice_idx

    #    raise ValueError('wrong index number{}'.format(index))

    #def __len__(self):
    #    return sum(self.count)

    def __getitem__(self, index):

        #study_idx, slice_idx = self._index(index)

        #image = self.images[study_idx][slice_idx]
        #mask = self.masks[study_idx][slice_idx]

        image = self.images[index]
        mask = self.masks[index]

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


#write_images('/home/baiyu/Downloads/promise12/TrainingData_Part1', 'test')


#datasets = []
#for training_list in glob.iglob('/home/baiyu/WorkSpace/vim_configure/training_list/*.txt'):
#
#
#    datasets.append(Promise12(training_list))
#
#datasets = torch.utils.data.ConcatDataset(datasets)
#
#import time
#
#start = time.time()
#for i in range(len(datasets)):
#    datasets[i]
#
#finish = time.time()
#print('{:0.2f}'.format(finish - start))
#print('average: {:0.2f}'.format(len(datasets) / (finish - start)))
#print(len(datasets))
#
##import sys
##print(sys.getsizeof(dataset))
##
##print(len(dataset))
##
##for i in range(len(dataset)):
##    dataset[i]
#
##for i in range(2):
##
##    i = random.choice(range(len(dataset)))
##    img, mask = dataset[i]
##    img = normalize(img)
##    mask = mask * 255
##
##    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##    cv2.drawContours(img, contours, -1, (255),1)
##    #cv2.imshow('img', img)
##    #cv2.imshow('mask', mask)
##    cv2.imwrite('img.jpg', img)
##    cv2.imwrite('mask.png', mask)
##    #cv2.waitKey(0)
##


#import transforms
#
#print(transforms.GaussianNoise.__init__)
#print(help(transforms.GaussianBlur))


#import torchvision.transforms as transforms
#trans = transforms.Compose([
#    transforms.ToPILImage(),
#    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
#])

#import glob
#from conf import settings
#
#datasets = []
#for list_path in glob.iglob(os.path.join(settings.TRAINING_LIST, '*.txt')):
#    print(list_path)
#    dataset = Promise12(list_path)
#    datasets.append(dataset)

#datasets = torch.utils.data.ConcatDataset(datasets)
#print(len(datasets))

#import utils

#print(utils.compute_mean_and_std(datasets))




#for i in range(60):
#    img, mask = dataset[i]
#    #print(img.size)
#    #print(mask.size)
#    print(img.shape)
#    print(mask.shape)
#    print(np.unique(mask))
#
#    #img.show()
#    #mask.show()
#    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(img, contours, -1, 255,1)
#
#    cv2.imshow('img', img)
#    cv2.imshow('mask', mask)
#    cv2.waitKey(0)