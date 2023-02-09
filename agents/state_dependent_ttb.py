from agents.ttb_sequential import TakeTheBestSequential
import numpy as np
from collections import defaultdict


class StateDependent(TakeTheBestSequential):
    """
    A tabular Q learning based algorithm that learns a Q value for every feature. This is used to inform the cue
    ordering used in the TTB heuristic
    """
    def __init__(self, num_features, current_state):
        super().__init__(num_features, current_state)
        self.Q = defaultdict(lambda: np.zeros(num_features))

    def learn(self, current_state, action_selection: list, rewards: list):
        """
        update the cue Q values for the present state
        """
        state = tuple(current_state)
        action_selection = np.array(action_selection)
        self.Q[state] *= (1 - self.alpha)
        for cue in range(self.num_features):
            max_cue_val = max(action_selection[:, cue])
            best_actions_ix = [a[cue] == max_cue_val for a in action_selection]
            best_actions = action_selection[best_actions_ix]
            best_actions_q = np.mean([max(self.Q[tuple(a)], default=0) for a in best_actions])
            reward = np.mean(np.array(rewards)[best_actions_ix])
            update = self.gamma * best_actions_q + reward
            self.Q[state][cue] += self.alpha * update

    def store_data(self, *args):
        pass
