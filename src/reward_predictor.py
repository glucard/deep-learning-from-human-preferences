from .reward_predictor_network import RewardPredictorNetwork
import torch as th
from random import sample
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import tkinter as tk
from .human_feedback_interface import interface_pick
from .utils import RunningStat

def save_video(frames, i):
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"{i+1}.avi", fourcc, 20.0, (80, 80), isColor=False)

    for frame in frames:
        # Write the frame to the video file
        out.write(frame[0])

    # Release the VideoWriter object
    out.release()

class RewardPredictor:
    def __init__(self, obs_shape: tuple, n_predictors: int, n_action:int, device: str=None):

        # check_device
        if not device:
            device = "cuda" if th.cuda.is_available() else "cpu"
        self.device = device

        self.n_action = n_action
        self.n_predictors = n_predictors
        self.predictors = [RewardPredictorNetwork(obs_shape, n_action, device).to(device) for _ in range(n_predictors)]
        self.D = []
        self.temp_experience = {
            "seq_obs": [],
            "seq_actions": [],
            "true_reward": [],
        }
        self.r_norm = RunningStat(shape=n_predictors)
    
    def predict(self, seq_obs:th.Tensor, seq_action:th.Tensor) -> th.Tensor:
        # tensors to device
        seq_obs = seq_obs.to(self.device)
        seq_action = seq_action.to(self.device)

        [model.eval() for model in self.predictors]
        # normalize
        with th.no_grad():
            preds = th.stack([p.forward(seq_obs.unsqueeze(0), seq_action.unsqueeze(0)) for p in self.predictors]).view(self.n_predictors, -1).cpu()
            # std, mean = th.std_mean(preds)
            # n_pred = (preds-mean) / std
            preds = preds.transpose(0,1)
            for p in preds:
                self.r_norm.push(p)
            preds -= self.r_norm.mean
            preds -= self.r_norm.mean
            preds /= (self.r_norm.std + 1e-12)
            preds *= 0.05
            # norm_preds = th.norm(preds, p=2, dim=-1) * 0.05
            # pred_reward = th.mean(norm_preds)
            preds = preds.transpose(0,1)
            pred_reward = th.mean(preds, dim=0)
            
        # todo : val loss: (1.1, 1.5) of train loss
        # todo : 10% human error


        return pred_reward[-1]
    
    def train(self) -> float:
        [model.train() for model in self.predictors]
        D = []
        for d in self.D:
            D.append(d) ##### optimizer later
        
        k = round(len(D) * 0.8)

        run_loss = 0
        for p in self.predictors:
            sampled_D = sample(D, k)
            # run_loss += p.train_rp(sampled_D)
            run_loss += p.train_rp(D)

        return run_loss / self.n_predictors

    def add_temp_experience(self, seq_obs, seq_actions, reward):
        self.temp_experience['seq_obs'].append(seq_obs)
        self.temp_experience['seq_actions'].append(seq_actions)
        self.temp_experience['true_reward'].append(reward)

    def reset_temp_experience(self):
        self.temp_experience['seq_obs'].clear()
        self.temp_experience['seq_actions'].clear()
        self.temp_experience['true_reward'].clear()

    def add_feedback(self, seg1:tuple[th.Tensor, th.Tensor], seg2:tuple[th.Tensor, th.Tensor], mu:tuple[float, float]):
        mu = th.tensor(mu, dtype=th.float32)
        self.D.append((seg1, seg2, mu))

    def get_syntetic_feedback(self, k:int) -> None:
        for i in range(k):
            segments = sorted(sample(range(len(self.temp_experience['seq_obs'])), 2))
            seq_obs = self.temp_experience['seq_obs'][segments[0]], self.temp_experience['seq_obs'][segments[1]]
            seq_actions = self.temp_experience['seq_actions'][segments[0]], self.temp_experience['seq_actions'][segments[1]]
            true_rewards = self.temp_experience['true_reward'][segments[0]], self.temp_experience['true_reward'][segments[1]]
            
            true_rewards = np.array(true_rewards)

            if true_rewards.sum() == 0.0:
                continue
            
            if true_rewards[0] == true_rewards[1]:
                self.add_feedback(seg1=(seq_obs[0],seq_actions[0]), seg2=(seq_obs[1], seq_actions[1]), mu=(.5,.5))
                continue
            
            better_seg = np.array(true_rewards).argmax()
            self.add_feedback(seg1=(seq_obs[0],seq_actions[0]), seg2=(seq_obs[1], seq_actions[1]), mu=(1.0, 0.0) if better_seg == 0 else (0.0, 1.0))

    def get_human_feedback(self, k:int) -> None:
        for i in range(k):

            segments = sorted(sample(range(len(self.temp_experience['seq_obs'])), 2))
            seq_obs = self.temp_experience['seq_obs'][segments[0]], self.temp_experience['seq_obs'][segments[1]]
            seq_actions = self.temp_experience['seq_actions'][segments[0]], self.temp_experience['seq_actions'][segments[1]]
            true_rewards = self.temp_experience['true_reward'][segments[0]], self.temp_experience['true_reward'][segments[1]]

            for i, observations in enumerate(seq_obs):
                save_video(np.array(observations.numpy()*255,dtype=np.uint8), i)
            
            human_feedback = interface_pick()
            
            segments = (seq_obs[0], seq_actions[0]), (seq_obs[1],seq_actions[1])

            if human_feedback == '1':
                self.add_feedback(seg1=segments[0], seg2=segments[1], mu=(1.0,.0))
                
            elif human_feedback == '2':
                self.add_feedback(seg1=segments[0], seg2=segments[1], mu=(.0,1.0))
                
            elif human_feedback == 'e':
                self.add_feedback(seg1=segments[0], seg2=segments[1], mu=(.5,.5))