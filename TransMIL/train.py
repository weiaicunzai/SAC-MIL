import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
import argparse

from datasets import WSIDataModule
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
    parser.add_argument('--gpus', default=0, required=True, type=int)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, default='mixer')
    args = parser.parse_args()
    return args

#---->main
def main(args):

    #---->Initialize seed
    pl.seed_everything(2023)

    #---->load loggers
    loggers = load_loggers('logs', args.dataset, args.fold)

    #---->load callbacks
    callbacks = load_callbacks('ckpt', args.dataset, args.fold)

    #---->Define Data
    data_root = os.path.join(os.path.dirname(__file__), 'dataset_csv')
    dataset_root = os.path.join(data_root, args.dataset)
    dm = WSIDataModule(fold=args.fold, dataset_path=dataset_root)

    #---->Define Model
    model = ModelInterface(model=args.model)

    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=loggers,
        callbacks=callbacks,
        max_epochs= 200,
        devices=[int(args.gpus)],
        precision='16-mixed',
        check_val_every_n_epoch=1,
    )

    #---->train or test
    if args.stage == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()

    #---->main
    main(args)
