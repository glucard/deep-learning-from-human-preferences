import numpy as np
import torch as th

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy

from src.reward_predictor import RewardPredictor
from src.enduro_env import EnvWrapper
from src.train_reward_predictor_callback import TrainRewardPredictorCallback

import gymnasium

def get_device():
    return "cuda" if th.cuda.is_available() else "cpu"

def main():
    
    env = gymnasium.make("ALE/Enduro-v5", obs_type="grayscale", full_action_space=True)
    device = get_device()
    reward_model = RewardPredictor((1,80,80), 5, n_action=env.action_space.n, device=device)
    
    env = EnvWrapper(env=env, reward_predictor=reward_model, seq_len=30)

    # Define the custom policy with normalized_images set to False
    policy_kwargs = dict(
        normalize_images=False
    )

    model = RecurrentPPO(CnnLstmPolicy, env, policy_kwargs=policy_kwargs,
                         n_steps=1024,
                         batch_size=64,
                         verbose=2,
                         learning_rate=1e-4)
    #print(model.policy)

    callback = TrainRewardPredictorCallback(reward_model)
    model.learn(100_000, callback=callback)
    model.save('model')
    
    env = gymnasium.make("ALE/Enduro-v5", obs_type="grayscale", render_mode="human", full_action_space=True)
    env = EnvWrapper(env=env, reward_predictor=reward_model, seq_len=30)

    obs, _ = env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, truncated, info = env.step(action)
        episode_starts = dones
        env.render()


if __name__ == "__main__":
    main()