import numpy as np
from ttb_sequential import TakeTheBestSequential


class TTB_Rollouts(TakeTheBestSequential):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)

    def store_data(self, action_rollout_returns: dict):
        """
        remeber action selected and return after x time steps (where x is the length of a roll-out)
        :param action_rollout_returns:
        :return:
        """
        action_features = action_rollout_returns.items()
        returns = action_rollout_returns.keys()
        self.xx = np.vstack([self.xx, action_features])
        self.yy = np.append(self.yy, returns)

    def learn(self):
        """
        learning the correlation between the feature value and the outcome
        This is a proxy for validity (in the multi-option case)
        No need to consider feature dependencies in this version of validity
        """
        for i in range(self.num_features):
            self.beta[i] = np.correlate(self.yy[:, i], self.xx)

    def feature_importance(self, state):
        feature_importance = np.argsort(np.argsort(self.beta))
        return feature_importance
