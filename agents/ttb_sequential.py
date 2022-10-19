import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


class TakeTheBestSequential:
    def __init__(self, num_features, current_state):
        # extract parameters
        self.num_features = num_features
        self.current_state = current_state

        # initialise learning feature importance
        self.xx = np.zeros([0, num_features])
        self.yy = np.zeros([0, 1])
        self.beta = np.zeros([num_features])

        # hyper-parameters
        self.gamma = 0.99  # discount factor

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

    def choose_action(self, actions):
        action_ix, action = self.ttb_action(self.feature_importance(self.current_state), actions)
        self.current_state = action
        return action_ix

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
        weighted_state_features = state*self.beta
        feature_importance = np.argsort(np.argsort(weighted_state_features))
        return feature_importance

    def ttb_action(self, feature_importance, actions):
        """
        implements the ttb heuristic for action selection

        :param feature_importance: a ranking of the importance of features
        :param actions: actions to choose from
        :return: the best action, according to TTB
        """
        for _ in range(self.num_features):
            best_feature = np.argmax(feature_importance)
            feature_values = [a[best_feature] for a in actions]
            actions_arg_sort = np.unique(feature_values, return_inverse=1)[1]  # order the actions by their best feature
            # if multiple actions are "best", keep looking, otherwise stop looking
            if np.count_nonzero(actions_arg_sort) == self.num_features-1:
                action_ix = np.argmin(actions_arg_sort)
                break
        else:
            action_ix = np.random.choice(actions.shape[0])
        return action_ix, actions[action_ix]
