import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import numpy as np
from ood_methods.Energy import Energy

EPS = 1e-5
SQRT2 = math.sqrt(2)
POW2_3O2 = math.pow(2, 1.5)

def calc_template(a, b, c, d):
    return (((a + b) * c) - (b * d))


def calc_statistics(x, y, a, b):
    ksei_square = math.pi / 8
    nu = a * (x + b)
    de = torch.sqrt(torch.relu(1 + ksei_square * a * a * y))
    return torch.sigmoid(nu / (de + EPS))


def get_ep_prior(theta_m, theta_s, fx):
    fx = fx.mean().squeeze()
    fxsq = fx * fx
    cm, cs = fx * theta_m, fxsq * theta_s

    e_1 = calc_statistics(cm, cs, 1, 0)
    e_2 = calc_statistics(cm, cs, 4 - 2 * SQRT2, math.log(SQRT2 + 1))
    e_3 = calc_statistics(cm, cs, 6 * (1 - 1 / POW2_3O2), math.log(POW2_3O2 - 1))

    _p_1 = calc_template(cm, cs, e_1, e_2)
    _p_2 = calc_template(cm, 2 * cs, e_2, e_3) ###
    s_0 = e_1
    s_1 = _p_1 / (s_0 * fx + EPS)
    s_2 = (cs * e_1 + calc_template(cm, cs, _p_1, _p_2)) / (s_0 * fxsq + EPS)

    theta_m, theta_s = s_1, torch.relu(s_2 - s_1 * s_1) + EPS
    theta_m = torch.clamp(theta_m, EPS, 5)
    theta_s = torch.clamp(theta_s, EPS, 5)
    del cm, cs, e_1, e_2, e_3, s_0, s_1, s_2
    return theta_m, theta_s


def merge_ep_prior_and_get_cavity(client_m, client_s):
    lmd = [(1/(s+EPS)) for s in client_s]
    theta_s = 1/(sum(lmd)+EPS)
    theta_m = theta_s * sum([m*l for m,l in zip(client_m, lmd)])
    theta_m = torch.clamp(theta_m, EPS, 5)
    theta_s = torch.clamp(theta_s, EPS, 5)
    cavities = []
    for i in range(len(client_m)):
        rest_lmd = lmd[:i] + lmd[i+1:]
        rest_m = client_m[:i] + client_m[i+1:]
        c_s = (1 / (sum(rest_lmd) + EPS))
        c_m = (c_s * sum([m*l for m,l in zip(rest_m, rest_lmd)]))
        cavities.append((c_m, c_s))
    
    return theta_m, theta_s, cavities


def remove_cavity(tm, ts, cm, cs):
    tb = tm / (ts + EPS)
    td = -0.5 / (ts + EPS)
    cb = cm / (cs + EPS)
    cd = -0.5 / (cs + EPS)
    qb = tb - cb
    qd = torch.relu(td - cd) + EPS
    qs = - 2  / (qd + EPS)
    qm = qb * qs
    qm = torch.clamp(qm, EPS, 5)
    qs = torch.clamp(qs, EPS, 5)
    return qm, qs


class FedNPClient:

    def __init__(self, args, dataset, data_index):
        self.args = args
        self.data_index = data_index
        self.train_loader = DataLoader(dataset,
                                       batch_size=args.local_bs,
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(data_index),
                                       drop_last=False)
        print(f"length of dataset: {len(data_index)}")

    # for training
    def train(self, net, npn_model, cavity, weight):
        net.train()
        npn_model.train()

        lr = self.args.lr
        local_ep = self.args.local_ep
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        optimizer_npn = torch.optim.SGD(npn_model.parameters(), lr=lr, weight_decay=self.args.weight_decay)

        cavity_mu, cavity_sigma = cavity
        with torch.no_grad():
            fx_all = []
            for _, (images, _) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                _, fx = net(images)
                fx_all.append(fx)
            
            mu, sigma = get_ep_prior(cavity_mu, cavity_sigma, torch.cat(fx_all))

        for _ in range(local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                outputs, fx = net(images)

                # flood
                scores = 1 * torch.logsumexp(outputs / 1, dim=1)
                
                threshold = torch.quantile(scores, self.args.threshold, interpolation='lower')
                below_mask = scores < threshold
                loss_weights = torch.full_like(scores, 1)
                loss_weights[below_mask] = weight

                kld_loss = torch.mean(-0.5 * ((1 + torch.log(torch.sqrt(sigma + EPS).mean()) - mu.mean() ** 2 - torch.sqrt(sigma + EPS).mean())))
                mu, sigma = npn_model((mu, sigma))
                loss_np = ((net.linear1.weight.reshape(-1) - mu) ** 2 / (2 * sigma + EPS)).mean()
                loss = criterion(outputs, labels)
                loss = (loss * loss_weights).mean()

                optimizer.zero_grad()
                optimizer_npn.zero_grad()

                (loss +  0.01 * (loss_np + kld_loss)).backward()

                optimizer.step()
                optimizer_npn.step()
                mu, sigma = get_ep_prior(cavity_mu, cavity_sigma, fx.detach())
    
        with torch.no_grad():
            fx_all = []
            for _, (images, _) in enumerate(self.train_loader):
                images = images.cuda()
                _, fx = net(images)
                fx_all.append(fx)
            mu, sigma = get_ep_prior(cavity_mu, cavity_sigma, torch.cat(fx_all))
            mu, sigma = remove_cavity(mu, sigma, cavity_mu, cavity_sigma)

        ood = Energy(net, self.args.device)
        scores = ood.eval(self.train_loader)
        score = np.mean(scores)

        return net.state_dict(), len(self.data_index), mu, sigma, score
