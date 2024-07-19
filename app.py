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
    
    env = gymnasium.make("ALE/Enduro-v5", obs_type="grayscale")
    device = get_device()
    reward_model = RewardPredictor((1,80,80), 5, n_action=env.action_space.n, device=device)
    
    env = EnvWrapper(env=env, reward_predictor=reward_model, seq_len=30)

    # Define the custom policy with normalized_images set to False
    policy_kwargs = dict(
        normalize_images=False
    )

    model = RecurrentPPO(CnnLstmPolicy, env, policy_kwargs=policy_kwargs
                         ,n_steps=1024,
                         verbose=2)

    callback = TrainRewardPredictorCallback(reward_model)
    model.learn(10_000, callback=callback)


if __name__ == "__main__":
    main()