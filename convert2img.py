import os
import glob
import SimpleITK as sitk
import numpy as np
import cv2




DATA_PATH = '/home/baiyu/Downloads/promise12'
IMAGE_PATH = '/home/baiyu/Downloads/promise12/images'

def normalize(img):
    img = img.astype(float)
    img = np.maximum(img,0) / img.max() * 255.0
    img = np.uint8(img)

    return img

for path in glob.iglob(os.path.join(DATA_PATH, '**', '*.mhd'), recursive=True):

    itkimage = sitk.ReadImage(path)
    mr_scan = sitk.GetArrayFromImage(itkimage)

    base_name = os.path.basename(path)

    for slc_idx, slc in enumerate(mr_scan):

        image_name = base_name.split('.')[0]


        if 'TrainingData_Part' in path:
            if 'segmentation' in path:
                image_name = "{}_slice{}_{}.png".format(image_name, slc_idx, 'train')
                slc *= 255
                slc = np.uint8(slc)
            else:
                image_name = "{}_slice{}_{}.jpg".format(image_name, slc_idx, 'train')
                slc = normalize(slc)

        else:
            image_name = "{}_slice{}_{}.jpg".format(image_name, slc_idx, 'test')
            slc = normalize(slc)


        save_path = os.path.join(IMAGE_PATH, image_name)

        cv2.imwrite(save_path, slc)




