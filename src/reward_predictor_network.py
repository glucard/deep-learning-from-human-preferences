import torch as th
from torch import nn
import torch.nn.functional as F

class RewardPredictorNetwork(nn.Module):
    def __init__(self, observation_shape:tuple, n_action:int, device:str, features_dim:int=256, learning_rate:float=1e-4, weight_decay:float=0.01, dropout:float=0.1) -> None:

        """
        observation_shape:
        features_dim:
        weight_decay: l2 regularization
        dropout: dropout
        """

        self.device = device
        super(RewardPredictorNetwork, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # self.dropout = dropout

        in_channels = observation_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(8,8), stride=(4,4)),
            nn.BatchNorm2d(16),
            # nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2)),
            nn.BatchNorm2d(32),
            # nn.Dropout(dropout),
            nn.LeakyReLU(),
            # nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1)),
            # nn.BatchNorm2d(64),
            # nn.Dropout(dropout),
            # nn.LeakyReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs_sample = th.zeros((observation_shape), dtype=th.float32)
            n_flatten = self.cnn(th.as_tensor(obs_sample[None])).shape[1]

        # self.linear = nn.Sequential(
        #     nn.Linear(n_flatten, features_dim),
        #     nn.Dropout(dropout),
        #     nn.LeakyReLU(),
        # )
        self.lstm = nn.LSTM(n_flatten, features_dim, num_layers=1)# , dropout=dropout) # + 1 for action
        self.lstm_relu = nn.LeakyReLU()
        self.output = nn.Sequential(
            nn.Linear(features_dim + n_action, 1),
        )


    def forward(self, seq_observation: th.Tensor, seq_action: th.Tensor) -> th.Tensor:
        
        seq_observation = seq_observation.to(self.device)
        seq_action = seq_action.to(self.device)
        
        batch_size, seq_len, c, h, w = seq_observation.shape

        # Reshape to process each frame through the CNN
        seq_observation = seq_observation.view(batch_size * seq_len, c, h, w)
        x = self.cnn(seq_observation)
        # x = self.linear(x)
        x = x.view(batch_size, seq_len, -1)
        #x = self.linear(x)
        x, (hn, cn) = self.lstm(x)
        x = F.dropout(self.lstm_relu(x), 0.2)
        x = th.concat([x, seq_action], dim=-1)
        x = self.output(x)
        return x
    
    def train_rp(self, D) -> float:
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer.zero_grad()
        run_loss = 0
        # todo : val loss
        
        experiences = []
        
        labels = []
        preds = []
        for d in D:
            (seq_obs_1, seq_a_1), (seq_obs_2, seq_a_2), mu = d # m is a tensor:(m1, m2)
            seq_obs_1 = seq_obs_1.to(self.device)
            seq_a_1 = seq_a_1.to(self.device)
            seq_obs_2 = seq_obs_2.to(self.device)
            seq_a_2 = seq_a_2.to(self.device)
            mu = mu.to(self.device)#.unsqueeze(0)

            # y_1 = self.forward(seq_obs_1, seq_a_1)
            # y_2 = self.forward(seq_obs_2, seq_a_2)
            y = self.forward(th.stack([seq_obs_1, seq_obs_2]), th.stack([seq_a_1, seq_a_2])).view(2, -1)

            labels.append(mu)
            preds.append(y.sum(dim=1))
            #y = th.stack([y_1, y_2], dim=1)
            #y = y.sum(dim=0)
            # print(y)
            
            #probs = F.softmax(y, dim=0)
            #log_probs = th.log(probs)
            #experiences.append(th.matmul(mu, log_probs))

        #loss = -th.stack(experiences).sum()
        preds = th.stack(preds)
        labels = th.stack(labels)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        run_loss += loss.item()

        self.optimizer.step()
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        # print("Gradient Norm:", total_norm)
        self.optimizer.zero_grad() # just to free memory
        return run_loss#  / len(D)

def debug():
    obs_shape = (3, 80, 80)

    seq_len = 5
    seq_obs_sample = th.randn((seq_len, *obs_shape), dtype=th.float32)
    seq_action_sample = th.randint(0, 3, (seq_len, 1))

    reward_predictor_net = RewardPredictorNetwork(obs_shape)
    print(reward_predictor_net(seq_obs_sample, seq_action_sample).shape)

if __name__ == "__main__":
    debug()