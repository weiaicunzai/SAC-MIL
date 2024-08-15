import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import sys

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--gpus', default=[2])
    parser.add_argument('--fold', default=0)
    parser.add_argument('--weights', default=None, type=str)

    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(2021)

    #---->load loggers
    loggers = load_loggers('logs')

    #---->load callbacks
    callbacks = load_callbacks(ckpt_path='ckpt_weights')

    #---->Define Data
    DataInterface_dict = {'train_batch_size': 1,
                'train_num_workers': 4,
                'test_batch_size': 1,
                'test_num_workers': 4,
                # 'dataset_name': cfg.Data.dataset_name,
                'fold': args.fold,
                'data_root': '/data/smb/syh/WSI_cls/cam16',
                'dataset_name': 'WSIDataset',
                # 'dataset_cfg': cfg.Data,
                }
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    ModelInterface_dict = {
                            'model': 'Mixer',
                            'log': loggers
                            }

    # print(cfg.log_path)
    # print(cfg.Data)
    model = ModelInterface(**ModelInterface_dict)

    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        # logger=cfg.load_loggers,
        logger=loggers,
        # callbacks=cfg.callbacks,
        callbacks=callbacks,
        # max_epochs= cfg.General.epochs,
        max_epochs=200,
        # gpus=cfg.General.gpus,
        devices=[int(args.gpus)],
        # amp_level=cfg.General.amp_level,
        # precision=cfg.General.precision,
        precision='16-mixed',
        # accumulate_grad_batches=cfg.General.grad_acc,
        # deterministic=True,
        check_val_every_n_epoch=1,
    )

    if args.stage == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        new_model = ModelInterface.load_from_checkpoint(checkpoint_path=args.weights, cfg=cfg)
        trainer.test(model=new_model, datamodule=dm)
        #model_paths = list(cfg.log_path.glob('*.ckpt'))
        #model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        #for path in model_paths:
        #    print(path)
        #    # new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
        #    trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    # cfg = read_yaml(args.config)

    #---->update
    # cfg.config = args.config
    # cfg.General.gpus = args.gpus
    # cfg.General.server = args.stage
    # cfg.Data.fold = args.fold

    #---->main
    main(args)
