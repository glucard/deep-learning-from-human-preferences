
import gymnasium
import numpy as np
from gymnasium import spaces
from PIL import Image

from reward_predictor import RewardPredictor


def preprocess_frame(frame):
    img = Image(frame).resize((80,80))
    return np.asarray(img, dtype=np.uint8)[None]

class EnvWrapper(gymnasium.Wrapper):
    def __init__(self, reward_predictor:RewardPredictor, seq_len:int=5):
        env = gymnasium.make("ALE/Enduro-v5", obs_type="grayscale")
        super(EnvWrapper, self).__init__(env)
        self.reward_predictor = reward_predictor
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(seq_len, 1, 80, 80), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = preprocess_frame(obs)
        return obs, info

    def step(self, action):
        obs, _reward, done, truncated, info = self.env.step(action)
        obs = preprocess_frame(obs)
        reward = self.reward_predictor(obs, actions)
        return obs, reward, done, truncated, info