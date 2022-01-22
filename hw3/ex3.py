import math
import random
import json

ids = ["311280283", "313535379"]


def euclidean(a, b):
    """
    Calculates euclidean distance between two points a(x1,y1) and b(x2,y2)
    :param a: (x1,y1)
    :param b: (x2,y2)
    :return: euclidean distance
    """
    return sum([(x - y) ** 2 for x, y in zip(a, b)]) ** 0.5


def manhattan(a, b):
    """
    Calculates manhattan distance between two points a(x1,y1) and b(x2,y2)
    :param a: (x1,y1)
    :param b: (x2,y2)
    :return: manhattan distance
    """
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


class DroneAgent:
    def __init__(self, n, m):
        actions = ['pick', 'move_up', 'move_down', 'move_left', 'move_right', 'deliver']
        self.n = n
        self.m = m
        self.mode = 'train'  # do not change this!
        self.q_learner = QLearning(actions, n, m)

    def select_action(self, obs0):
        return self.select_best_action(obs0)

    def select_best_action(self, obs0):
        """
        Selects the best action possible, if there are no more package to deliver so we return 'reset', else we call
        the choose_action method of our q_learner
        :param obs0: state
        :return: best action possible
        """
        if not obs0['packages']:
            return 'reset'
        for k, v in obs0.items():
            if type(v) == set:
                obs0[k] = list(v)
        return self.q_learner.choose_action(obs0, self.mode)

    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def update(self, obs0, action, obs1, reward):
        """
        Updates our q_learner q values
        :param obs0: previous state
        :param action: action
        :param obs1: current state
        :param reward: reward given by applying action on previous state which led to current state
        """
        for k, v in obs0.items():
            if type(v) == set:
                obs0[k] = list(v)
        for k, v in obs1.items():
            if type(v) == set:
                obs1[k] = list(v)
        self.q_learner.learn(obs0, action, reward, obs1)


class QLearning:
    def __init__(self, actions, n, m, epsilon=0.9, alpha=0.32, gamma=0.85):
        self.q = {}
        self.n = n
        self.m = m
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # learning constant
        self.gamma = gamma  # discount constant
        self.actions = actions

    def get_q(self, state, action):
        return self.q.get((state, action), 0)

    def choose_action(self, state, mode):
        """
        Chooses the best action possible. If we can deliver or pick we will do so. Otherwise, if mode is train then
        randomly pick any feasible action from action space with probability of self.epsilon (0.9). Else find the best
        action with respect to q value (action which maximize q value)
        :param state: current state
        :param mode: train/eval
        :return: best action
        """
        action_space = self.drone_action_builder(state)
        if action_space == 'deliver':
            return 'deliver'
        elif action_space == 'pick':
            return 'pick'
        state = json.dumps(state)
        if mode == 'train' and random.random() < self.epsilon:
            for action in action_space:
                if (state, action) not in self.q:
                    return action
            return random.choice(action_space)

        q = [self.get_q(state, a) for a in action_space]
        maxQ = max(q)
        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(action_space)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)
        action = action_space[i]
        return action

    def drone_action_builder(self, state):
        """
        Builds all possible actions from a given state
        :param state: game state
        :return: if the drone can deliver or pick then 'deliver' / 'pick' is returned as a string.
        otherwise a list of possible actions is returned where each action is represented by a string
        """

        action_space = []
        x, y = state['drone_location']
        packages = {pack[0]: pack[1] for pack in state['packages'] if type(pack[1]) == tuple}
        drone_packages = [val[1] for val in state['packages'] if type(val[1]) != tuple]
        target_loc = state['target_location']

        # -------------------------- add deliver --------------------------
        if drone_packages and (x, y) == target_loc:
            return 'deliver'

        # -------------------------- add pick --------------------------
        if len(drone_packages) < 2:
            for pack_coord in packages.values():
                if (x, y) == pack_coord:
                    return 'pick'
        # -------------------------- add moves and wait --------------------------

        # move up
        if x - 1 >= 0:
            # legal position
            # add movement
            action_space.append('move_up')

        # move down
        if x + 1 < self.n:
            # legal position
            # add movement
            action_space.append('move_down')

        # move left
        if y - 1 >= 0:
            # legal position
            # add movement
            action_space.append('move_left')

        # move right
        if y + 1 < self.m:
            # legal position
            # add movement
            action_space.append('move_right')
        return action_space

    def learn(self, state1, action1, reward, state2):
        """
        Q-learning:
            Q(s, a) += alpha * (reward_func(s,a) + max(Q(s')) - Q(s,a))
        """
        if action1 == 'pick' and reward > 0:
            reward = 40
        state1 = json.dumps(state1)
        state2 = json.dumps(state2)
        q_max = max([self.get_q(state2, a) for a in self.actions])
        old_q = self.q.get((state1, action1), None)
        if old_q is None:
            self.q[(state1, action1)] = reward
        else:
            self.q[(state1, action1)] = old_q + self.alpha * (reward + self.gamma * q_max - old_q)
