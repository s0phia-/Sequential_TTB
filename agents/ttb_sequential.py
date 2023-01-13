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
            action_ix = random.choice(range(len(actions)))
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
        original_actions = actions
        actions = np.unique(actions, axis=0)
        feature_importance = np.array(feature_importance, dtype=float)
        for i in range(self.num_features):
            # TODO add in something for feature direction
            best_feature = np.nanargmax(feature_importance)
            feature_values = [a[best_feature] for a in actions]
            if np.sum(feature_values == max(feature_values)) == 1:  # if there is 1 action with the max feature value
                action_ix = np.argmax(feature_values)  # select that action. Otherwise keep looping through features
                break
            else:  # find the next best deciding feature, by deleting the old best. Only keep actions which were best
                actions = actions[np.argwhere(feature_values == np.max(feature_values)).flatten()]

                feature_importance[best_feature] = np.nan
        else:
            action_ix = np.random.choice(actions.shape[0])
        best_action = actions[action_ix]
        whole_list_action_ix = int(np.where(np.all(original_actions == best_action, axis=1))[0][0])
        return whole_list_action_ix, best_action
