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
            best_action = np.argmax(actions_transpose[i]) # TODO there might be a few!
            action_rollout_returns = best_rollout_returns[best_action]
            if action_rollout_returns == 1:
                rollout_feature_validities[i] = 1
        self.xx = np.vstack([self.xx, rollout_feature_validities])

    def learn(self, window_size=5000):
        x = self.xx
        # use sliding window
        if len(x) > window_size:
            x = x[-window_size:, ]

        total = x.shape[0]
        rr = np.sum(x, axis=0)
        self.beta = rr/total


class TTB_Roll_Cond_Val(TTB_Rollouts):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)

    def store_data(self, rollout_actions: list, rollout_returns: list):
        """

        :param rollout_actions:
        :param rollout_returns:
        :return:
        """
        # make a binary vector showing whether the rollout outcome was best (1) or not (0)
        max_rollout_return = max(rollout_returns)
        best_rollout_returns = np.where(rollout_returns == max_rollout_return, 1, 0)

        # does this action have the highest feature value for any features?
        max_feature_values = np.max(rollout_actions, axis=0)
        feature_is_max = np.where(rollout_actions==max_feature_values, 1, 0)

        self.xx = np.r_[self.xx, feature_is_max]
        self.yy = np.append[self.yy, best_rollout_returns]

    def learn(self, window_size=5000):
        """
        find feature validities using equation rr/(rr+ww) where ww is wrong decisions and rr is correct decisions.
        This equation for validity is from https://www.researchgate.net/publication/23753521_One-reason_decision_making

        :param window_size: a moving window so only the most recent data is used
        :return: a vector weighting features by their conditional validity
        """
        x = self.xx
        y = self.yy

        # implement moving window
        if len(x) > window_size:
            x = x[-window_size:,]
            y = y[-window_size:,]

        # find feature validities
        conditional_feature_val = []
        for feature in self.num_features:
            rr = np.sum(
                np.take(x, np.argwhere(np.array(y) == 1).flatten(), axis=0),
                axis=0)
            ww = np.sum(
                np.take(x, np.argwhere(np.array(y) == 0).flatten(), axis=0),
                axis=0)
            validity = rr / (rr+ww)

            # get best feature index
            best_feature = np.argmax(validity)
            conditional_feature_val.append(best_feature)

            # remove best feature from calculation
            x[:, best_feature] = 0

        # assign features values in order of validities
        self.beta = np.argsort(-np.array(conditional_feature_val))
