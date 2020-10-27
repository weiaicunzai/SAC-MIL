import os
import glob
import random



IMAGE_PATH = '/home/baiyu/Downloads/promise12/images'

images = glob.glob(os.path.join(IMAGE_PATH, '*train.jpg'))
random.shuffle(images)

splits = [images[x : 1+x+len(images)//5] for x in range(0, len(images), 1 + len(images) // 5)]

for file_idx in range(5):

    with open(str(file_idx) + '.txt', 'w') as f:

        split = splits[file_idx]

        for image_path in split:
            f.write(image_path + '\n')









