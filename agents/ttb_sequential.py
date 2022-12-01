import numpy as np
import abc
import random


class TakeTheBestSequential(abc.ABC):
    def __init__(self, num_features, current_state):
        # extract parameters
        self.num_features = num_features
        self.current_state = current_state

        # initialise learning feature importance
        self.xx = np.zeros([0, num_features])
        self.yy = np.zeros([0, 1])
        self.beta = np.random.rand(num_features)

        # hyper-parameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 0.9

    @abc.abstractmethod
    def store_data(self, *args):
        pass

    def choose_action(self, actions):
        if random.random() > self.epsilon:
            action_ix = random.choice(range(len(actions)))  # Todo
            action = actions[action_ix]
        else:
            action_ix, action = self.ttb_action(self.feature_importance(self.current_state), actions)
        self.current_state = action
        return action_ix

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def feature_importance(self, state):
        pass

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
