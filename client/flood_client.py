import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ood_methods.Energy import Energy

class FLoodClient:

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
        net.train()

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss(reduction='none')
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        for _ in range(local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs, _ = net(images)
                # get ood score
                # Energy
                scores = 1 * torch.logsumexp(outputs / 1, dim=1)
                
                threshold = torch.quantile(scores, self.args.threshold, interpolation='lower')
                below_mask = scores < threshold
                loss_weights = torch.full_like(scores, 1)
                loss_weights[below_mask] = weight

                loss = criterion(outputs, labels)
                loss = (loss * loss_weights).mean()
                loss.backward()
                optimizer.step()

        ood = Energy(net, self.args.device)
        scores = ood.eval(self.train_loader)
        score = np.mean(scores)

        return net.state_dict(), len(self.data_index), score
