import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from agents.ttb_sequential import TakeTheBestSequential


class TtbQValues(TakeTheBestSequential):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)

    def store_data(self, state, state_prime, reward, actions_prime, done):
        """
        Update dataset and targets. Target updates as reward + gamma*q(s')
        :param state: current state
        :param state_prime: action selected
        :param reward: reward received from environment
        :param actions_prime: possible after_states from state_prime
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
        x = self.xx
        y = self.yy
        # if len(y) > 10000:
        #     x = x[-10000:, ]
        #     y = y[-10000:]
        reg = LinearRegression(fit_intercept=False).fit(x, y)
        self.beta = reg.coef_

    def feature_importance(self, state):
        """
        Learn the importance of features for a state where the most important feature has the highest value of
        feature weight * feature value

        :param state: a state to learn feature values for
        :return: a list with the relative importance of feature i in position i
        """
        feature_importance = np.argsort(np.argsort(self.beta))
        # [3 4 1 5 2 6 7 0] 20
        return feature_importance



