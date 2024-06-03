import os
from datetime import datetime
import math

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

DATA_ROOT = '/data/smb/syh/WSI_cls/'

cam16_max_len = 44380
cam16_num_seg = 256
cam16_wsi_len = math.ceil(cam16_max_len / cam16_num_seg) * cam16_num_seg