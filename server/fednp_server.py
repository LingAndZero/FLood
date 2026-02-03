import copy
import torch
from timm.utils import CheckpointSaver
import numpy as np
import math

from utils.dataset import *
from utils.model import *
from torch.utils.data import DataLoader



class NPNLinearLite(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels, dual_input = True, init_type = 0):
        super(NPNLinearLite, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dual_input = dual_input

        self.W_m = torch.nn.Parameter(2 * math.sqrt(6) / math.sqrt(in_channels + out_channels) * (torch.rand(in_channels, out_channels) - 0.5))
        self.bias_m = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if self.dual_input:
            x_m, x_s = x
        else:
            x_m = x
            x_s = x.clone()
            x_s = 0 * x_s

        o_m = torch.mm(x_m.unsqueeze(0), self.W_m)
        o_m = o_m + self.bias_m.expand_as(o_m)

        o_s = torch.mm(x_s.unsqueeze(0), self.W_m * self.W_m)

        return o_m.squeeze(0), o_s.squeeze(0)


class NPNSigmoid(torch.nn.Module):
    def __init__(self):
        super(NPNSigmoid, self).__init__()
        self.xi_sq = math.pi / 8
        self.alpha = 4 - 2 * math.sqrt(2)
        self.beta = - math.log(math.sqrt(2) + 1)

    def forward(self, x):
        assert(len(x) == 2)
        o_m, o_s = x
        a_m = torch.sigmoid(o_m / (1 + self.xi_sq * o_s) ** 0.5)
        a_s = torch.sigmoid(self.alpha * (o_m + self.beta) / (1 + self.xi_sq * self.alpha ** 2 * o_s) ** 0.5) - a_m ** 2
        return a_m, a_s


class NPN(torch.nn.Module):
    def __init__(self, args, model):
        super(NPN, self).__init__()
        self.model = model
        dim = self.model.linear1.weight.numel()
        self.net = torch.nn.Sequential(
            NPNLinearLite(args, 10, 10, True),
            NPNSigmoid(),
            NPNLinearLite(args, 10, dim, True)
        )
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, x):
        return self.net(x)


class FedNPServer:
    
    def __init__(self, args, dataset_test, checkpoint_logdir):
        self.args = args

        print("Load the test dataset")
        self.dataset = dataset_test
        self.test_loader = DataLoader(self.dataset, batch_size=args.bs, shuffle=False, drop_last=False)
        
        print("Initialize the global model")
        self.model_g = get_model(args)
        self.model_g.train()

        self.npn_models = [NPN(args, get_model(args)).to(args.device) for i in range(args.num_users)]
        self.cavities = [(torch.zeros(10).cuda(), torch.ones(10).cuda() / (args.num_users - 1)) for i in range(args.num_users)]


    # Select clients
    def select_clients(self):
        m = max(int(self.args.fraction * self.args.num_users), 1)
        clients = np.random.choice(range(self.args.num_users), m, replace=False)
        return clients

    # For aggregate
    def aggregate(self, local_models, local_n_train, local_mu, local_sigma, local_score, clients):
        weights = np.array(local_n_train) / sum(local_n_train)
        scores = np.array(local_score) / sum(local_score)
        combined_weights = weights + self.args.alpha * scores
        weights = combined_weights / sum(combined_weights)

        w_glob = copy.deepcopy(local_models[0])
        global_state = self.model_g.state_dict()

        for param_name in global_state:
            global_param = global_state[param_name]
            param_shape = global_param.shape

            flat_global = global_param.view(-1)

            weighted_sum = torch.zeros_like(flat_global, dtype=torch.float32)
            for model_idx, local_model in enumerate(local_models):
                local_param = local_model[param_name].view(-1)
                weighted_sum.add_((local_param - flat_global) * weights[model_idx])

            w_glob[param_name] = (weighted_sum + flat_global).view(param_shape)

        self.model_g.load_state_dict(w_glob)

        _, _, cavities = merge_ep_prior_and_get_cavity(local_mu, local_sigma)

        for i, idx in enumerate(clients):
            self.cavities[idx] = cavities[i]

    # for test
    def test(self, epoch):
        self.model_g.eval()

        accuracy = 0
        with torch.no_grad():
            for _, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                outputs, _ = self.model_g(images)
                accuracy += (outputs.argmax(1) == labels).sum()

        return accuracy.item() / len(self.dataset)
    

EPS = 1e-5
SQRT2 = math.sqrt(2)
POW2_3O2 = math.pow(2, 1.5)


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