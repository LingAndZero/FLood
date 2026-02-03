import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class GEN:

    def __init__(self, model,device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
        '''
        self.gamma = 0.1
        self.M = 10

    @ torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in data_loader:
            images = images.to(self.device)
            output, _ = self.model(images)
            smax = (F.softmax(output, dim=1)).data.cpu().numpy()
            probs_sorted = np.sort(smax, axis=1)[:,-self.M:]
            scores = np.sum(probs_sorted ** self.gamma * (1 - probs_sorted) ** self.gamma, axis=1)

            result.append(-scores)

        return np.concatenate(result)