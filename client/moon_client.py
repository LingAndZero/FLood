import torch
import copy
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from ood_methods.Energy import Energy


class MoonClient:

    def __init__(self, args, dataset, data_index, previous_model):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        print(f"length of dataset: {len(data_index)}")
        self.previous_model = copy.deepcopy(previous_model).cpu()

    # for training
    def train(self, net, weight):
        global_model = copy.deepcopy(net).to(self.args.device)
        self.previous_model.to(self.args.device)
        net.train()

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        cos = nn.CosineSimilarity(dim=-1)
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        for _ in range(local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs1, r1 = net(images)

                scores = 1 * torch.logsumexp(outputs1 / 1, dim=1)
                threshold = torch.quantile(scores, self.args.threshold, interpolation='lower')
                below_mask = scores < threshold
                loss_weights = torch.full_like(scores, 1)
                loss_weights[below_mask] = weight

                _, r2 = global_model(images)
                posi = cos(r1, r2)
                logits = posi.reshape(-1, 1)

                _, r3 = self.previous_model(images)
                nega = cos(r1, r3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                logits /= self.args.temperature
                targets = torch.zeros(images.size(0)).to(self.args.device).long()

                loss1 = criterion(outputs1, labels)
                loss = (loss1 * loss_weights).mean()
                loss2 = self.args.mu * criterion(logits, targets)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

        self.previous_model = copy.deepcopy(net).cpu()

        ood = Energy(net, self.args.device)
        scores = ood.eval(self.train_loader)
        score = np.mean(scores)
        
        return net.state_dict(), len(self.data_index), score
