import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import utils
from model import Model


class SupervisedNet(nn.Module):
    """
    Supervised model using the same ResNet-18 backbone but with a linear classification
    head (512 -> 10), which I attached to the backbone output. Here labeled data is used.
    """
    def __init__(self, num_class=10):
        super(SupervisedNet, self).__init__()

        #same backbone as SimCLR (ResNet-18 modified for CIFAR-10)
        self.f = Model().f

        #Classification head: maps 512-dim backbone output to 10 class scores, no projector haed
        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        #flatten output
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


def train_val(net, data_loader, train_optimizer):
    """
    Train/evaluate the model for each epoch.
    If train_optimizer provided -> training mode, otherwise evaluation
    """
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    data_bar = tqdm(data_loader)

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # Forward pass: compute class scores
            out = net(data)

            # Cross-entropy loss: standard loss for classification tasks
            loss = loss_criterion(out, target)

            if is_train:
                #update weights (backbone + classifier)
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)

            #top-1 and top-5 accuracy
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()

            data_bar.set_description(
                '{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                .format('Train' if is_train else 'Test',
                        epoch, epochs,
                        total_loss / total_num,
                        total_correct_1 / total_num * 100,
                        total_correct_5 / total_num * 100)
            )

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised Learning Baseline')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for Adam optimizer')

    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epochs

    #Load CIFAR-10 with standard transforms and use labeled data 
    train_data = CIFAR10(root='data', train=True,
                         transform=utils.train_transform, download=False)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    test_data = CIFAR10(root='data', train=False,
                        transform=utils.test_transform, download=False)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    
    model = SupervisedNet(num_class=len(train_data.classes)).cuda()

    #same settings as SimCLR 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    #Cross-entropy loss for multi-class classification
    loss_criterion = nn.CrossEntropyLoss()

    results = {
        'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
        'test_loss': [], 'test_acc@1': [], 'test_acc@5': []
    }

    if not os.path.exists('results'):
        os.mkdir('results')

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        #Train for one epoch
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)

        #Evaluate on test set
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        #Save results to CSV 
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/supervised_statistics.csv', index_label='epoch')

        #save best model
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/supervised_model.pth')

    print(f'Best Test Accuracy: {best_acc:.2f}%')