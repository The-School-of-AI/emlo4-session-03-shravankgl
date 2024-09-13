import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path


def test_epoch(model, data_loader):
    # write code to test this epoch
    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    out = {"Test loss": test_loss, "Accuracy": accuracy}
    print(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # create MNIST test dataset and loader

    # create model and load state dict

    # test epoch function call

    with (Path(args.save_dir) / "model" / "eval_results.json").open("w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
