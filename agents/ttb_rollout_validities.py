import numpy as np
from agents.ttb_rollouts import TTB_Rollouts
from scipy.stats import pearsonr


class TTB_Roll_Val(TTB_Rollouts):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)

    def store_data(self, rollout_actions: list, rollout_returns: list):
        rollout_feature_validities = np.zeros([self.num_features])
        max_rollout_return = max(rollout_returns)
        best_rollout_returns = np.where(rollout_returns == max_rollout_return, 1, 0)
        actions_transpose = np.transpose(np.array(rollout_actions))
        for i in range(self.num_features):
            # find the action with the highest value for feature i
            best_action = np.argmax(actions_transpose[i])
            action_rollout_returns = best_rollout_returns[best_action]
            if action_rollout_returns == 1:
                rollout_feature_validities[i] = 1
        self.xx = np.vstack([self.xx, rollout_feature_validities])

    def learn(self, window_size=5000):
        x = self.xx
        y = self.yy
        if len(y) < 20:  # must be length >=2 for correlation calc
            return
        # use sliding window
        if len(y) > window_size:
            x = x[-window_size:, ]
            y = y[-window_size:]

        total = x.shape[0]
        print(total)
        rr = np.sum(x, axis=0)
        self.beta = rr/total
        print(self.beta)
