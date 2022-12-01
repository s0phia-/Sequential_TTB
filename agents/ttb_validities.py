import numpy as np
from agents.ttb_sequential import TakeTheBestSequential
from statistics import mean


class Ttb_validities(TakeTheBestSequential):
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)

    def store_data(self, action, available_actions, reward):
        if reward == 0 or len(available_actions == 1):
            return
        selected_action = available_actions[action]
        print(available_actions)
        unselected_actions = available_actions[:action] + available_actions[action+1:]
        feature_validity = np.zeros([1, self.num_features])
        for i in range(self.num_features):
            unselected_avg = mean([x[i] for x in unselected_actions])
            if selected_action[i] > unselected_avg:
                feature_validity[i] = 1
        self.xx = np.vstack([self.xx, feature_validity])

    def learn(self):
        data = np.transpose(self.xx)
        self.beta = [sum(data[i]) for i in range(len(data))]

    def feature_importance(self, state):
        feature_importance = np.argsort(np.argsort(self.beta))
        return feature_importance
