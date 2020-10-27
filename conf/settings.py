import os
from datetime import datetime

#voc2012 bgr

# camvid bgr
MEAN = (0.13499225)
STD = (0.12254033)

CHECKPOINT_FOLDER = 'checkpoints'
LOG_FOLDER ='runs'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

DATA_PATH = '/content/drive/My Drive/dataset/camvid'

IMAGE_SIZE = (480, 360)

MILESTONES = [100, 150]

IGNORE_LABEL = 255

EPOCHS = 200


TRAINING_LIST = '/home/baiyu/WorkSpace/vim_configure/training_list'

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
