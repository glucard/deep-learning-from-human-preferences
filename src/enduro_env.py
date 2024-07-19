
import gymnasium
import numpy as np
import torch as th
from gymnasium import spaces
from PIL import Image

from .reward_predictor import RewardPredictor
from collections import deque

def preprocess_obs(frame):
    img = Image.fromarray(frame).resize((80,80))
    return np.asarray(img, dtype=np.uint8)[None]

class EnvWrapper(gymnasium.Wrapper):
    def __init__(self, env:gymnasium.Env, reward_predictor:RewardPredictor, seq_len:int=5):

        super(EnvWrapper, self).__init__(env)
        self.seq_len = seq_len
        self.reward_predictor = reward_predictor
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, 80, 80), dtype=np.uint8)

    def reset(self, **kwargs):
        self.seq_actions = np.zeros((self.seq_len, self.action_space.n), dtype=np.float32)
        self.seq_obs = np.zeros((self.seq_len, 1, 80, 80), dtype=np.float32)

        obs, info = self.env.reset(**kwargs)
        obs = preprocess_obs(obs)
        # obs_0
        self.seq_obs[-1] = obs

        return obs, info

    def step(self, action):
        obs, _reward, done, truncated, info = self.env.step(action)
        
        # pre process obs
        obs = preprocess_obs(obs)
        
        # get reward
        # action_t
        # one hot action
        action_one_hot = np.zeros((self.action_space.n), dtype=np.float32)
        action_one_hot[action] = 1
        self.seq_actions[:-1] = self.seq_actions[1:]
        self.seq_actions[-1] = action_one_hot

        # to tensor
        seq_obs = th.tensor(self.seq_obs, dtype=th.float32)
        seq_actions = th.tensor(self.seq_actions, dtype=th.float32)

        # reward_t
        predicted_reward = self.reward_predictor.predict(seq_obs, seq_actions)

        self.reward_predictor.add_temp_experience(seq_obs, seq_actions, _reward)
        
        # obs_t+1
        self.seq_obs[:-1] = self.seq_obs[1:]
        self.seq_obs[-1] = obs
        return obs, predicted_reward, done, truncated, info