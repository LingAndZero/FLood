import torch
from torch import nn
from torch.utils.data import DataLoader
import copy


class SCAFFOLDClient:

    def __init__(self, args, dataset, data_index, global_variate):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        # self.local_variate = copy.deepcopy(global_variate)
        self.local_variate = {k: v.detach().cpu().clone() for k, v in global_variate.items()}
        print(f"length of dataset: {len(data_index)}")

    def _move_variate_to_device(self, variate_dict):
        device = self.args.device
        return {k: v.to(device, non_blocking=True) for k, v in variate_dict.items()}
    
    def _state_dict_to_cpu(self, sd):
        return {k: v.detach().cpu().clone() for k, v in sd.items()}
    
    # for training
    def train(self, net, global_variate, weight):
        net.train()

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        g_var = self._move_variate_to_device(global_variate)
        l_var = self._move_variate_to_device(self.local_variate)

        for _ in range(local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                outputs, _ = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for name, p in net.named_parameters():
                        if name in g_var and name in l_var:
                            p.add_( -lr * (g_var[name] - l_var[name]) )
                # w = net.state_dict()
                # for key in w:
                #     if 'bias' in key or 'weight' in key:
                #         w[key] -= lr * (global_variate[key] - self.local_variate[key])
                # net.load_state_dict(w)

        w_backup = copy.deepcopy(net.state_dict())
        # w_backup = self._state_dict_to_cpu(net.state_dict())

        for _, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            optimizer.zero_grad()
            outputs, _ = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        k = len(self.train_loader)
        w_after = net.state_dict()
        
        # for key in self.local_variate:
        #     self.local_variate[key] = (w_backup[key] - w_after[key]) / (k * lr)
        
        new_local_var = {}
        with torch.no_grad():
            for key in self.local_variate.keys():
                new_local_var[key] = (w_backup[key].to(self.args.device) - w_after[key]).div_(k * lr).detach().cpu()
        self.local_variate = new_local_var


        return w_backup, len(self.data_index), self.local_variate
