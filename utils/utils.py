import os
from datetime import datetime

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

def load_loggers(log_path):

    log_path = os.path.join(log_path, TIME_NOW)
    print(f'---->Log dir: {log_path}')

    #---->TensorBoard
    # tb_logger = pl_loggers.TensorBoardLogger(log_path+str(log_name),
    #                                          name = version_name, version = f'fold{cfg.Data.fold}',
    #                                          log_graph = True, default_hp_metric = False)
    # print(hasattr(tb_logger, 'experiments'))
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(log_path)

    # return [tb_logger, csv_logger]
    return [csv_logger]

#---->load Callback
def load_callbacks(ckpt_path):

    Mycallbacks = []
    # Make output path
    output_path = os.path.join(ckpt_path, TIME_NOW)
    os.makedirs(output_path, exist_ok=True)

    Mycallbacks.append(ModelCheckpoint(monitor = 'val_acc',
                                         dirpath = ckpt_path,
                                         filename = '{epoch:02d}-{val_loss:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'min',
                                         save_weights_only = True))
    return Mycallbacks