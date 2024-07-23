from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from .reward_predictor import RewardPredictor


class TrainRewardPredictorCallback(BaseCallback):
    def __init__(self, rp: RewardPredictor, verbose=0):
        super(TrainRewardPredictorCallback, self).__init__(verbose)
        self.rp = rp
        self.true_rewards = []
        self.pred_rewards = []

    def _on_step(self) -> bool:
        true_reward = self.locals.get("infos", None)[-1].get("true_reward", None)
        pred_reward = self.locals.get("infos", None)[-1].get("pred_reward", None)
        self.true_rewards.append(true_reward)
        self.pred_rewards.append(pred_reward)
        
        return True

    def _on_rollout_end(self) -> None:
        
        ep_true_reward = sum(self.true_rewards)
        ep_pred_reward = sum(self.pred_rewards)
        self.logger.record("custom/true_reward", ep_true_reward)
        self.logger.record_mean("custom/true_reward_mean", ep_true_reward)
        self.logger.record("custom/pred_reward", ep_pred_reward)

        self.true_rewards.clear()
        self.pred_rewards.clear()

        k = int(len(self.rp.temp_experience['seq_obs']) * 0.01) # 1%
        k = k if k > 1 else 2
        self.rp.get_human_feedback(k)
        if len(self.rp.D) > 1:
            for i in range(10):
                loss = self.rp.train()
                print(f"Reward Predict loss = {loss}\n")
        self.rp.reset_temp_experience()

        
        self.logger.record("custom/D", len(self.rp.D))