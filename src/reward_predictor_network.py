import torch as th
from torch import nn
import torch.nn.functional as F

class RewardPredictorNetwork(nn.Module):
    def __init__(self, observation_shape:tuple, n_action:int, device:str, features_dim:int=256, learning_rate:float=3e-4, weight_decay:float=.0, dropout:float=.0) -> None:

        """
        observation_shape:
        features_dim:
        weight_decay: l2 regularization
        dropout: dropout
        """
        self.device = device
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
        self.lstm = nn.LSTM(features_dim + n_action, features_dim, num_layers=1, dropout=dropout) # + 1 for action
        self.lstm_relu = nn.ReLU()
        self.output = nn.Sequential(
            nn.Linear(features_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.optimizer = th.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, seq_observation: th.Tensor, seq_action: th.Tensor) -> th.Tensor:
        seq_observation = seq_observation.to(self.device)
        seq_action = seq_action.to(self.device)
        x = seq_observation
        x = self.cnn(x)
        x = self.linear(x)
        x = th.concat([x, seq_action], dim=-1) # concat obs and action ?? <--|||||||||||||||-------------|||||||||||||||---------------|||||||||||||||---------
        x, (hn, cn) = self.lstm(x)
        x = self.lstm_relu(x)[-1]
        x = self.output(x)
        return x
    
    def train_rp(self, D) -> float:
        self.optimizer.zero_grad()
        run_loss = 0
        # todo : val loss
        for d in D:
            (seq_obs_1, seq_a_1), (seq_obs_2, seq_a_2), m = d # m is a tensor:(m1, m2)
            seq_obs_1 = seq_obs_1.to(self.device)
            seq_a_1 = seq_a_1.to(self.device)
            seq_obs_2 = seq_obs_2.to(self.device)
            seq_a_2 = seq_a_2.to(self.device)
            m = m.to(self.device)
            y_1 = self.forward(seq_obs_1, seq_a_1)
            y_2 = self.forward(seq_obs_2, seq_a_2)
            y = th.stack([y_1, y_2])
            # print(F.softmax(y, dim=0), F.softmax(y, dim=1).shape)
            # print(m.unsqueeze(0), m.unsqueeze(0).shape)
            loss = th.matmul(m.unsqueeze(0), F.softmax(y, dim=0))
            run_loss += loss.item()
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad() # just to free memory
        return run_loss / len(D)

def debug():
    obs_shape = (3, 80, 80)

    seq_len = 5
    seq_obs_sample = th.randn((seq_len, *obs_shape), dtype=th.float32)
    seq_action_sample = th.randint(0, 3, (seq_len, 1))

    reward_predictor_net = RewardPredictorNetwork(obs_shape)
    print(reward_predictor_net(seq_obs_sample, seq_action_sample).shape)

if __name__ == "__main__":
    debug()