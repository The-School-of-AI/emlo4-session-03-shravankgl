import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch.multiprocessing as mp


def train(rank, args, model, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)
    # training code for mnist hogwild


def main():
    parser = argparse.ArgumentParser(description="MNIST Training Script")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        metavar="N",
        help="how many training processes to use (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # create model and setup mp

    model.share_memory()

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }

    # create mnist train dataset

    # mnist hogwild training process
    processes = []


# save model ckpt


if __name__ == "__main__":
    main()
