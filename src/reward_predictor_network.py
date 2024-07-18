import torch as th
from torch import nn
import torch.nn.functional as F

class RewardPredictorNetwork(nn.Module):
    def __init__(self, input_shape, weight_decay):
        super(RewardPredictorNetwork, self).__init__()

        self.cnn = nn.Sequential(
            
        )