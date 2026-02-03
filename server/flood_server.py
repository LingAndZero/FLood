import copy
import torch
from timm.utils import CheckpointSaver
import numpy as np

from utils.dataset import *
from utils.model import *
from torch.utils.data import DataLoader


class FLoodServer:
    
    def __init__(self, args, dataset_test, checkpoint_logdir):
        self.args = args

        print("Load the test dataset")
        self.dataset = dataset_test
        self.test_loader = DataLoader(self.dataset, batch_size=args.bs, shuffle=False, drop_last=False)
        
        print("Initialize the global model")
        self.model_g = get_model(args)
        self.model_g.train()


    # Select clients
    def select_clients(self):
        m = max(int(self.args.fraction * self.args.num_users), 1)
        clients = np.random.choice(range(self.args.num_users), m, replace=False)
        return clients

    # For aggregate
    def aggregate(self, local_models, local_n_train, local_score):
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