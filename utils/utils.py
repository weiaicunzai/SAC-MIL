import os
from datetime import datetime

from pytorch_lightning.callbacks import ModelCheckpoint


DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers
def load_loggers(log_root, dataset_name, fold):

    # log_path = cfg.General.log_path
    # Path(log_path).mkdir(exist_ok=True, parents=True)
    # log_name = Path(cfg.config).parent
    # version_name = Path(cfg.config).name[:-5]
    # cfg.log_path = Path(log_path) / log_name / version_name / f'fold{cfg.Data.fold}'

    # DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    # #time of we run the script
    # TIME_NOW = datetime.now().strftime(DATE_FORMAT)

    log_path = os.path.join(log_root, dataset_name, fold, TIME_NOW)
    os.makedirs(log_path)

    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(log_path+str(log_path),
                                             log_graph = True, default_hp_metric = False)
    print(hasattr(tb_logger, 'experiments'))
    #---->CSV
    # csv_logger = pl_loggers.CSVLogger(log_path)

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
