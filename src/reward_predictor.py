from .reward_predictor_network import RewardPredictorNetwork
import torch as th
from random import sample
import matplotlib.pyplot as plt

class RewardPredictor:
    def __init__(self, obs_shape: tuple, n_predictors: int, device: str=None):

        # check_device
        if not device:
            device = "cuda" if th.cuda.is_available() else "cpu"
        self.device = device

        self.n_predictors = n_predictors
        self.predictors = [RewardPredictorNetwork(obs_shape).to(device) for _ in range(n_predictors)]
        self.D = []
    
    def predict(self, seq_obs, seq_action) -> th.Tensor:
        # tensors to device
        seq_obs = seq_obs.to(self.device)
        seq_action = seq_action.to(self.device)

        # normalize
        with th.no_grad():
            preds = th.stack([p.forward(seq_obs, seq_action) for p in self.predictors])
            norm_preds = th.norm(preds)
            pred_reward = th.mean(norm_preds)
        
        # todo : val loss: (1.1, 1.5) of train loss
        # todo : 10% human error

        return pred_reward
    
    def train(self) -> float:
        D = []
        for d in self.D:
            D.append(d[0].to(self.device), d[1].to(self.device), d[2].to(self.device)) ##### optimizer later
        
        k = round(len(D) * 0.8)

        run_loss = 0
        for p in self.predictors:
            sampled_D = sample(D, k)
            run_loss += p.train_rp(sampled_D)

        return run_loss / self.n_predictors
    
    def add_human_feedback(self, seg1:tuple[th.Tensor, th.Tensor], seg2:tuple[th.Tensor, th.Tensor], mu:tuple[float, float]):
        mu = th.tensor(mu, dtype=th.float32)
        self.D.append(seg1, seg2, mu)

    def get_human_feedback(self, rollout:tuple[tuple[th.tensor, th.tensor]], k:int) -> None:
        for i in range(k):
            segments = sample(rollout, 2)
            for i, s in enumerate(segments):
                observations = s[0]
                print(f"segment {i+1}:")
                for obs in observations:
                    plt.imshow(obs)
                    plt.show()
                print("\n")

            acceptable_feedbacks = ['1', '2', 'e', 'n']
            while True:
                human_feedback = input("Which segment do you prefer? (1: (1better), 2:(2better), e:(equal good) or n(incomparable))")
                if human_feedback in acceptable_feedbacks:
                    break
                print("invalid answer. Try again.", end=" ")

            if human_feedback == '1':
                self.add_human_feedback(seg1=segments[0], seg2=segments[1], mu=(1.0,.0))
                
            elif human_feedback == '2':
                self.add_human_feedback(seg1=segments[0], seg2=segments[1], mu=(.0,1.0))
                
            elif human_feedback == 'e':
                self.add_human_feedback(seg1=segments[0], seg2=segments[1], mu=(.5,.5))