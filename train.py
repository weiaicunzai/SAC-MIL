import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
import argparse

from dataset import WSIDataModule
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
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--feat', type=str, required=True)
    args = parser.parse_args()
    print(args)
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
    data_root = '/data/smb/syh/WSI_cls/'

    if args.feat not in ['uni', 'res50']:
        raise ValueError(
            'wrong value'
        )

    if args.feat == 'uni':
        feat_ext = 'feat_clam_uni'
    elif args.feat == 'res50':
        feat_ext = 'feat_clam_resnet50'

    dataset_root = os.path.join(data_root, args.dataset, feat_ext)
    dm = WSIDataModule(fold=args.fold, dataset_path=dataset_root, dataset=args.dataset)

    #---->Define Model
    model = ModelInterface(model=args.model, feat=args.feat, debug=True)

    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0,
        logger=loggers,
        callbacks=callbacks,
        max_epochs=200,
        devices=[int(args.gpus)],
        precision='16-mixed',
        check_val_every_n_epoch=1,
    )

    #---->train or test
    if args.stage == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        new_model = ModelInterface.load_from_checkpoint(args.ckpt, feat=args.feat, debug=False)
        trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()

    #---->main
    main(args)
