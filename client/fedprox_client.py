import torch
import copy
from torch import nn
from torch.utils.data import DataLoader


class FedProxClient:

    def __init__(self, args, dataset, data_index):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        print(f"length of dataset: {len(data_index)}")

    # for training
    def train(self, net, weight):
        global_model = copy.deepcopy(net).to(self.args.device)
        global_weight_collector = list(global_model.parameters())
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

                # For fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((self.args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

        return net.state_dict(), len(self.data_index), _
