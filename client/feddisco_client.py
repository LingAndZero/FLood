import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from scipy import special


class FedDiscoClient:

    def __init__(self, args, dataset, data_index):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        self.difference = get_distribution_difference(args, dataset, data_index)

    # for training
    def train(self, net, weight):
        net.train()

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        for _ in range(local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs, _ = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return net.state_dict(), len(self.data_index), self.difference


def get_distribution_difference(args, dataset, data_index):
    global_dist = np.ones(args.num_classes) / args.num_classes
    local_dist = np.zeros(args.num_classes)

    for index in data_index:
        local_dist[dataset[index][1]] += 1

    local_dist = local_dist / len(data_index)
    difference = special.kl_div(local_dist, global_dist)
    difference = np.sum(difference)
    
    return difference
