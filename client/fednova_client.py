import torch
from torch import nn
import copy
from torch.utils.data import DataLoader


class FedNovaClient:

    def __init__(self, args, dataset, data_index):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        self.iteration = (self.args.local_ep * len(self.data_index)) // self.args.local_bs
        self.rho = 0.9

    # for training
    def train(self, net, weight):
        net.train()
        global_weights = copy.deepcopy(net.state_dict())

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

        coeff = (self.iteration - self.rho * (1 - pow(self.rho, self.iteration)) / (1 - self.rho)) / (1 - self.rho)
        state_dict = net.state_dict()
        norm_grad = copy.deepcopy(global_weights)
        for key in norm_grad:
            norm_grad[key] = torch.div(global_weights[key] - state_dict[key], coeff)

        return norm_grad, len(self.data_index), coeff
