# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import sys
sys.path.append(os.getcwd())

import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import MlpMixer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_utils import get_world_size

# from conf.settings import co
from conf import settings


logger = logging.getLogger(__name__)

import torchmetrics
metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(num_classes=2, average = 'macro', task='MULTICLASS'),
                torchmetrics.AUROC(num_classes=2, average = 'macro', task='MULTICLASS'),
                torchmetrics.F1Score(num_classes=2, average = 'macro', task='MULTICLASS'),
            ]
        )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 2

    if args.dataset_name == 'cam16':
        wsi_len = settings.cam16_wsi_len
        print(wsi_len)

    model = MlpMixer(
        input_dim=1024,
        hidden_dim=512,
        seq_len=wsi_len,
        token_dim=256,
        channel_dim=512,
        num_tokens=196,
        num_classes=num_classes
    )
    print(model)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    preds = []
    gts = []
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            # print(logits.shape)
            prob = torch.nn.functional.softmax(logits, dim=-1)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds.append(prob)

            gts.append(y)


        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    val_res = metrics.to(args.device)(preds, gts)
    metrics.reset()

    print('Validation: ', val_res)

    return val_res['MulticlassAccuracy']

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_root", default='/data/smb/syh/WSI_cls/')
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_type", choices=["Mixer-B_16", "Mixer-L_16",
                                                 "Mixer-B_16-21k", "Mixer-L_16-21k"],
                        default="Mixer-B_16",
                        help="Which model to use.")
    parser.add_argument("--ckpt_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=512, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.img_size = 224


    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    train_loader, test_loader = get_loader(args)


    model.load_state_dict(torch.load(args.ckpt_dir))

    valid(args, model, test_loader)



if __name__ == "__main__":
    main()
