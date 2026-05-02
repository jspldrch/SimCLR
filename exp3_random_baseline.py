import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import utils
from model import Model


class RandomNet(nn.Module):
    """
    Backbone with randomly initialized weights that are never trained.
    This serves as a lower-bound baseline to show what accuracy is
    achievable with pure random features, without any learning.
    """
    def __init__(self, num_class=10):
        super(RandomNet, self).__init__()

        # Load backbone with random weights (no pretrained weights)
        self.f = Model().f

        # Immediately freeze all backbone weights
        # The backbone will never be updated during training
        for param in self.f.parameters():
            param.requires_grad = False

        # Linear classification head — only this will be trained
        self.fc = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        with torch.no_grad():
            x = self.f(x)
            feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    data_bar = tqdm(data_loader)

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
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
    batch_size = 256
    epochs = 100

    train_data = CIFAR10(root='data', train=True,
                         transform=utils.test_transform, download=False)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    test_data = CIFAR10(root='data', train=False,
                        transform=utils.test_transform, download=False)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model with random weights
    model = RandomNet(num_class=len(train_data.classes)).cuda()

    # Only train the linear classification head
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()

    results = {
        'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
        'test_loss': [], 'test_acc@1': [], 'test_acc@5': []
    }

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)

        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/random_baseline_statistics.csv', index_label='epoch')

        if test_acc_1 > best_acc:
            best_acc = test_acc_1

    print(f'Best Test Accuracy (Random Baseline): {best_acc:.2f}%')