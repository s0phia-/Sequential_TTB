import numpy as np
import random
from copy import deepcopy


class ChasingGridWorld:
    """
    Gridworld with a monster that chases the agent with some probability, behaves randomly the rest of the time
    """

    def __init__(self, cols, rows):
        self.num_cols = cols
        self.num_rows = rows
        self.agent_position, self.monster_position = self.reset()
        self.num_features = 4
        self.state = self.get_state()
        self.random_monster = 0
        self.loss_reward = -100

    def reset(self):
        """
        set the agent position and monster position to random (distinct) locations on the grid
        :return: agent and monster's new positions
        """

        def random_grid_position(class_self):  # get a random grid position
            x, y = random.randint(0, class_self.num_rows - 1), random.randint(0, class_self.num_cols - 1)
            return np.array((float(x), float(y)))

        self.agent_position = random_grid_position(self)
        self.monster_position = random_grid_position(self)

        while (self.monster_position == self.agent_position).all():  # ensure monster and agent positions are not same
            self.monster_position = random_grid_position(self)

        return self.agent_position, self.monster_position

    def step(self, action: list):
        """
        Move the agent according to the action, then move the monster to chase the agent. Game finishes if the agent
        and mosnter are ever on the same square
        :param action: the [x, y] directions the agent should move in
        :return: the state with the agent's and monster's new position
        """
        def check_action_valid(which_agent, act, class_self=self):
            if which_agent == "agent":
                position = class_self.agent_position
            elif which_agent == "monster":
                position = class_self.monster_position
            else:
                return "invalid which_agent"
            # check whether the action moves the agent/monster outside the bounds of the grid
            if (position + np.array(act, dtype=int) >= [class_self.num_rows, class_self.num_cols]).any() \
                    or (position + np.array(act, dtype=int) < [0, 0]).any():
                # agent is trying to move outside the bounds of the grid
                return False
            else:
                return True

        # only move the agent if the action is valid. Otherwise, action does nothing
        if check_action_valid("agent", action) is True:
            self.agent_position += action

        if (self.agent_position == self.monster_position).all():  # if the move takes the agent into the monster
            done = True  # game is over
            reward = self.loss_reward
            state = self.get_state()
            return state, reward, done, ""

        monster_action = self.get_monster_action()

        # only move monster if action is valid. Otherwise action does nothing
        if check_action_valid("monster", monster_action) is True:
            self.monster_position += np.array(monster_action, dtype=int)
        state = self.get_state()

        # if the agent and monster share a location, the game is over
        if (self.agent_position == self.monster_position).all():
            done = True
            reward = self.loss_reward
        else:
            done = False
            reward = 1

        return state, reward, done, ""

    def get_monster_action(self):
        if random.uniform(0, 1) < self.random_monster:  # if the monster behaves randomly
            monster_action_x = random.randint(0, 1)
            monster_action_y = 1 - monster_action_x
            monster_action = (monster_action_x, monster_action_y)

        else:  # monster chases the agent - it moves 1 square towards the agent on the axis with the biggest difference
            diff = self.agent_position - self.monster_position
            biggest_diff_axis = np.argmax([abs(diff)])
            monster_action = [0, 0]
            monster_action[biggest_diff_axis] = diff[biggest_diff_axis] / abs(diff[biggest_diff_axis])  # gets +/- 1
        return monster_action

    def render(self):
        """
        Print a grid with 0s everywhere except 1 in the agent's position and 2 in the monster's position
        """
        grid = np.zeros([self.num_rows, self.num_cols])
        grid[self.agent_position[0], self.agent_position[1]] = 1
        grid[self.monster_position[0], self.monster_position[1]] = 2
        print(grid)

    def get_state(self):
        """
        Calculate the current state. The four features are
        agent_x - monster_x
        agent_y - monster_y
        monster_x - agent_x
        monster_y - agent_y
        :return: current state
        """
        state = np.zeros(self.num_features)
        # state[0:2] = self.agent_position - self.monster_position
        # state[2:4] = self.monster_position - self.agent_position
        state[0:2] = self.agent_position
        state[2:4] = self.monster_position
        state = tuple(state)
        self.state = state
        return state


class ChasingGridWorldAfterStates(ChasingGridWorld):
    def __init__(self, cols, rows):
        super().__init__(cols, rows)
        self.afterstate_mapping = None

    def get_after_states(self, include_terminal=False):
        possile_actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        afterstates = {}
        rewards = []
        agent_position, monster_position = tuple(self.agent_position), tuple(self.monster_position)
        for a in possile_actions:
            afterstate, reward, done, _ = self.step(a)
            if include_terminal or not done:
                afterstates[tuple(a)] = afterstate
                rewards.append(reward)
            self.agent_position, self.monster_position = np.array(agent_position), np.array(monster_position)
        self.afterstate_mapping = afterstates

        return list(afterstates.values()), rewards

    def get_current_state_features(self):
        return self.get_state()

    def get_action_from_afterstate(self, afterstate):
        """
        Given afterstate features, find the action that lead to it. If multiple after_states, return one at random
        :return: an action that lead to the given afterstate
        """
        actions = [k for k, v in self.afterstate_mapping.items() if (v == afterstate).all()]
        action = random.choice(actions)
        return action

    def single_rollout(self, action, policy_function, length):
        reset_agent_position, reset_monster_position = deepcopy(self.agent_position), deepcopy(self.monster_position)
        if (self.monster_position == self.agent_position).all():
            return self.loss_reward
        _, _, done, _ = self.step(action)
        if done:
            self.agent_position, self.monster_position = reset_agent_position, reset_monster_position
            return self.loss_reward
        rollout_return = 0
        for _ in range(length-1):
            action = policy_function(self.get_after_states(include_terminal=True)[0])
            _, reward, done, _ = self.step(action)
            rollout_return += reward
            if done:
                rollout_return = self.loss_reward
                break
        self.agent_position, self.monster_position = reset_agent_position, reset_monster_position
        return rollout_return

    def perform_rollouts(self, actions, policy_function, length=5, n=5):
        rollout_actions = []
        rollout_returns = []
        for action in range(len(actions)):
            action_all_returns = []
            for i in range(n):  # do n rollouts,
                action_all_returns.append(self.single_rollout(action, policy_function, length))
            rollout_actions.append(actions[action])
            rollout_returns.append(np.mean(action_all_returns))  # save the mean of the rollout returns
        return rollout_actions, rollout_returns
