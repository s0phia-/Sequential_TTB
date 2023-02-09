import numpy as np
from scipy.stats import pearsonr
from agents.ttb_sequential import TakeTheBestSequential


class TtbRolloutsCorrelation(TakeTheBestSequential):
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


class TtbRolloutsSummingScore(TakeTheBestSequential):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)

    def store_data(self, rollout_actions: list, rollout_returns: list):
        """
        Store results of the rollouts.

        The y vector (outcomes) are stored in binary format, with a 1 in position i if outcome i was (joint) best

        :param rollout_actions: The action features at the beginning of the rollout
        :param rollout_returns: The corresponding returns at the end of the rollout
        """

        rollout_actions = np.array(rollout_actions)
        # does this action have the highest feature value for any features?
        max_feature_values = np.max(rollout_actions, axis=0)
        #feature_is_max = np.where(rollout_actions == max_feature_values)

        for feature_ix in range(self.num_features):
            ttb_actions_feat_i = np.argwhere(rollout_actions[:, feature_ix] == max_feature_values[feature_ix]).flatten()
            mean_best_action_values = np.mean(np.take(rollout_returns, ttb_actions_feat_i))
            self.beta[feature_ix] += mean_best_action_values


        # self.xx = np.r_[self.xx, feature_is_max]
        # self.yy = np.append(self.yy, rollout_returns)

    def learn(self, window_size=5000000000000):
        """

        :param window_size: a moving window so only the most recent data is used
        :return:
        """
        # x = self.xx
        # y = self.yy
        #
        # # implement moving window
        # if len(x) > window_size:
        #     x = x[-window_size:, ]
        #     y = y[-window_size:, ]
        #
        # feature_value = []
        # for feature_ix in range(self.num_features):
        #     # sum the rollout outcomes for actions feature i would have selected using TTB
        #     sum_tx = np.sum(
        #         np.take(y, np.argwhere(np.array(x[feature_ix]) == 1).flatten(), axis=0), axis=0)
        #     feature_value.append(sum_tx)
        #
        # self.beta = feature_value
        pass
