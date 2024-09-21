import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# def train_epoch(epoch, args, model, device, data_loader, optimizer):
#     model.train()
#     for batch_idx, (data, target) in enumerate(data_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(data_loader.dataset),
#                 100. * batch_idx / len(data_loader), loss.item()))
#             if args.dry_run:
#                 break

# def test_epoch(model, device, data_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in data_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(data_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(data_loader.dataset),
#         100. * correct / len(data_loader.dataset)))
    
# def main():
#     # Parser to get command line arguments
#     parser = argparse.ArgumentParser(description='MNIST Training Script')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=2, metavar='N',
#                         help='number of epochs to train (default: 14)')
#     parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--no-mps', action='store_true', default=False,
#                         help='disables macOS GPU training')
#     parser.add_argument('--dry-run', action='store_true', default=False,
#                         help='quickly check a single pass')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--save-model', action='store_true', default=False,
#                         help='For Saving the current Model')
#     parser.add_argument("--resume", action='store_true', default=False, 
#                         help="Path to checkpoint to resume from")
    
#     args = parser.parse_args()
#     use_cuda = torch.cuda.is_available()
#     torch.manual_seed(args.seed)
#     device = torch.device("cuda" if use_cuda else "cpu")

#     # TODO: Load the MNIST dataset for training and testing
#     train_kwargs = {'batch_size': args.batch_size}
#     test_kwargs = {'batch_size': args.test_batch_size}

#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     dataset1 = datasets.MNIST('../data', train=True, download=True,
#                        transform=transform)
#     dataset2 = datasets.MNIST('../data', train=False,
#                        transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
#     model = Net().to(device)

#     # TODO: Add a way to load the model checkpoint if 'resume' argument is True
#     # TODO: Choose and define the optimizer here
#     start_epoch = 1
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#     #if args.resume:
#     if os.path.isfile("./model/mnist_cnn.pt"):
#         print("Loading model_checkpoint")
#         checkpoint = torch.load("./model/mnist_cnn.pt")
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#         model.load_state_dict(checkpoint["model_state_dict"])
#         #start_epoch = checkpoint["epoch"] + 1
#     else:
#         print("No model checkpoint found. Starting from scratch.")
   

#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
#     # TODO: Implement the training and testing cycles
#     for epoch in range(start_epoch, args.epochs + 1):
#         train_epoch(epoch, args, model, device, train_loader, optimizer)
#         test_epoch(model, device, test_loader)
#         scheduler.step()

#         # Hint: Save the model after each epoch
#     checkpoint = {
#     #"epoch": args.epochs,
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),}
#     torch.save(checkpoint, "./model/mnist_cnn.pt")


# if __name__ == "__main__":
#     main()
