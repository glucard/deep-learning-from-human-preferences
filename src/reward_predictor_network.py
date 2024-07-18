import torch as th
from torch import nn
import torch.nn.functional as F

class RewardPredictorNetwork(nn.Module):
    def __init__(self, observation_shape:tuple, features_dim:int=256, learning_rate:float=3e-4, weight_decay:float=.0, dropout:float=.0) -> None:

        """
        observation_shape:
        features_dim:
        weight_decay: l2 regularization
        dropout: dropout
        """
        super(RewardPredictorNetwork, self).__init__()

        # self.learning_rate = learning_rate
        # self.weight_decay = weight_decay
        # self.dropout = dropout

        in_channels = observation_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=(8,8), stride=(4,4)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(4,4), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs_sample = th.zeros((observation_shape), dtype=th.float32)
            n_flatten = self.cnn(th.as_tensor(obs_sample[None])).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(features_dim + 1, features_dim, num_layers=1, dropout=dropout) # + 1 for action
        self.lstm_relu = nn.ReLU()

        self.optimizer = th.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, seq_observation: th.Tensor, seq_action: th.Tensor) -> th.Tensor:
        x = seq_observation
        x = self.cnn(x)
        x = self.linear(x)
        x = th.concat([x, seq_action], dim=-1) # concat obs and action ?? <--|||||||||||||||-------------|||||||||||||||---------------|||||||||||||||---------
        x = self.lstm_relu(x)
        return x
    
    def train_rp(self, D):
        self.optimizer.zero_grad()

        for d in D:
            (seq_obs_1, seq_a_1), (seq_obs_2, seq_a_2), m = d # m is a tensor:(m1, m2)
            y_1 = self.forward(seq_obs_1, seq_a_1)
            y_2 = self.forward(seq_obs_2, seq_a_2)
            y = th.stack(y_1, y_2)
            loss = F.softmax(y, dim=1).dot(m.T)
            loss.backward()

        self.optimizer.step()

def debug():
    obs_shape = (3, 80, 80)

    seq_len = 5
    seq_obs_sample = th.randn((seq_len, *obs_shape), dtype=th.float32)
    seq_action_sample = th.randint(0, 3, (seq_len, 1))

    reward_predictor_net = RewardPredictorNetwork(obs_shape)
    reward_predictor_net(seq_obs_sample, seq_action_sample)

if __name__ == "__main__":
    debug()