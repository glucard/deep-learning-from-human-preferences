from stable_baselines3.common.callbacks import BaseCallback
from .reward_predictor import RewardPredictor

class TrainRewardPredictorCallback(BaseCallback):
    def __init__(self, rp: RewardPredictor, verbose=0):
        super(TrainRewardPredictorCallback, self).__init__(verbose)
        self.rp = rp

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        print("training")
        k = int(len(self.rp.temp_experience['seq_obs']) * 0.01) # 1%
        k = k if k > 1 else 2
        self.rp.get_human_feedback(k)
        if len(self.rp.D) > 1:
            for i in range(100):
                loss = self.rp.train()
                print(f"Reward Predict loss = {loss}\n")
        self.rp.reset_temp_experience()
        
        print("training end")