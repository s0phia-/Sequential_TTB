import numpy as np
from scipy.stats import pearsonr
from agents.ttb_sequential import TakeTheBestSequential


class TTB_Rollouts(TakeTheBestSequential):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)

    def store_data(self, rollout_actions: list, rollout_returns: list):
        """
        remember action selected and return after x time steps (where x is the length of a roll-out)
        :param rollout_returns:
        :param rollout_actions:
        :return:
        """
        nonzero_return_ix = np.nonzero(rollout_returns)

        action_features = np.array(rollout_actions)[nonzero_return_ix]
        returns = np.array(rollout_returns)[nonzero_return_ix]

        self.xx = np.vstack([self.xx, action_features])
        self.yy = np.append(self.yy, returns)

    def learn(self):
        """
        learning the correlation between the feature value and the outcome
        This is a proxy for validity (in the multi-option case)
        No need to consider feature dependencies in this version of validity
        """
        x = self.xx
        y = self.yy
        if len(y) < 2:  # must be length >=2 for correlation calc
            return
        # use sliding window
        if len(y) > 1000:
            x = x[-1000:, ]
            y = y[-1000:]
        for i in range(self.num_features):
            self.beta[i] = pearsonr(y, x[:, i])[0]  # returns correlation ceof and p val, only want corr


