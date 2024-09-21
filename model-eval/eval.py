import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
from model import Net


def test(args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    return test_epoch(model, device, test_loader)

def test_epoch(model, device, data_loader):
    # write code to test this epoch
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    out = {
        "Test loss": round(test_loss, 4),
        "Correct": correct,
        "Total": len(data_loader.dataset),
        "Accuracy": round(100. * correct / len(data_loader.dataset), 0)
        }
    print(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")
    parser.add_argument(
        '--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)'
    )
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
    test_dataset = datasets.MNIST(os.path.join(args.save_dir, 'data'), train=False, download=False, transform=transform)

    device = torch.device("cpu")
    model = Net().to(device)

    # create model and load state dict 
    model_checkpoint_path = os.path.join(args.save_dir, "model/mnist_cnn.pt")
    if os.path.isfile(model_checkpoint_path):
        print("Loading model_checkpoint")
        model.load_state_dict(torch.load(model_checkpoint_path))

        # test epoch function call
        kwargs = {'batch_size': args.test_batch_size}
        eval_results = test(args, model, device, test_dataset, kwargs)

        with (Path(args.save_dir) / "model" / "eval_results.json").open("w") as f:
            json.dump(eval_results, f)

    else:
        print("No model checkpoint found. test failed.")

    


if __name__ == "__main__":
    main()
