import os
from datetime import datetime

from pytorch_lightning.callbacks import ModelCheckpoint


DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers
def load_loggers(log_root, dataset_name, fold):

    log_path = os.path.join(log_root, dataset_name, fold, TIME_NOW)
    print('log_path', log_path)
    os.makedirs(log_path, exist_ok=True)

    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(log_path,
                                             log_graph = True, default_hp_metric = False)
    print(hasattr(tb_logger, 'experiments'))

    return [tb_logger]


#---->load Callback
def load_callbacks(ckpt_root, dataset_name, fold):

    Mycallbacks = []

    ckpt_path = os.path.join(ckpt_root, dataset_name, fold, TIME_NOW)
    os.makedirs(ckpt_path, exist_ok=True)


    Mycallbacks.append(ModelCheckpoint(monitor = 'val_acc',
                                         dirpath = ckpt_path,
                                         filename = '{epoch:02d}-{val_acc:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'max',
                                         save_weights_only = True))
    return Mycallbacks
