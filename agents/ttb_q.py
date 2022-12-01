import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from agents.ttb_sequential import TakeTheBestSequential


class Ttb_Q_values(TakeTheBestSequential):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)

    def store_data(self, state, state_prime, reward, actions_prime, done):
        """
        Update dataset and targets. Target updates as reward + gamma*q(s')
        :param state: current state
        :param state_prime: action selected
        :param reward: reward received from environment
        :param actions_prime: possible actions from state_prime
        """
        # update target
        if done:
            target = reward
        else:
            a_prime = self.ttb_action(self.feature_importance(state_prime), actions_prime)[1]
            q_prime = np.dot(a_prime, self.beta)
            target = reward + self.gamma * q_prime
        # update X and y
        self.xx = np.vstack([self.xx, state])
        self.yy = np.append(self.yy, target)

    def learn(self):
        """
        learn weight that, when multiplied by state features, indicate the cue ordering for choosing an action
        """
        reg = LinearRegression(fit_intercept=False).fit(self.xx, self.yy)
        self.beta = reg.coef_

    def feature_importance(self, state):
        """
        Learn the importance of features for a state where the most important feature has the highest value of
        feature weight * feature value

        :param state: a state to learn feature values for
        :return: a list with the relative importance of feature i in position i
        """
        weighted_state_features = self.beta
        feature_importance = np.argsort(np.argsort(weighted_state_features))
        return [5, 2, 1, 6, 7, 3, 0, 4]
        # return feature_importance

