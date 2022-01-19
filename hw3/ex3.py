import math
import random
import json

ids = ["311280283", "313535379"]


def euclidean(a, b):
    return sum([(x - y) ** 2 for x, y in zip(a, b)]) ** 0.5


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


class DroneAgent:
    def __init__(self, n, m):
        actions = ['pick', 'move_up', 'move_down', 'move_left', 'move_right', 'deliver']
        self.n = n
        self.m = m
        self.mode = 'train'  # do not change this!
        self.q_learner = QLearning(actions, n, m)
        self.t = 0
        self.episodes = 0

    def select_action(self, obs0):
        return self.select_best_action(obs0)

    def select_best_action(self, obs0):
        # print(f'my steps: {self.t}, my episodes: {self.episodes}')
        # action_space = self.drones_action_builder(obs0)
        if not obs0['packages']:
            # print('----------------------------')
            self.episodes += 1
            self.t = 0
            return 'reset'
        self.t += 1
        if self.t > 30:
            self.episodes += 1
            self.t = 1
        for k, v in obs0.items():
            if type(v) == set:
                obs0[k] = list(v)
        return self.q_learner.choose_action(obs0, self.mode, self.episodes)

    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def update(self, obs0, action, obs1, reward):
        # print(f'Last loc: {obs0["drone_location"]}  ---- Current loc: {obs1["drone_location"]}')
        for k, v in obs0.items():
            if type(v) == set:
                obs0[k] = list(v)
        for k, v in obs1.items():
            if type(v) == set:
                obs1[k] = list(v)
        self.q_learner.learn(obs0, action, reward, obs1)


class QLearning:
    def __init__(self, actions, n, m, epsilon=0.9, alpha=0.3, gamma=0.7):
        self.q = {}
        self.n = n
        self.m = m
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # learning constant
        self.gamma = gamma  # discount constant
        self.actions = actions

    def get_q(self, state, action):
        return self.q.get((state, action), 0)

    def choose_action(self, state, mode, episode):
        # if mode == 'train':
        #     if episode <= 133e3:
        #         self.epsilon = 0.5
        #     else:
        #         self.epsilon = 0.25
        # else:
        #     self.epsilon = 0.2
        action_space = self.drone_action_builder(state)
        # print(state)
        # print(action_space, state['drone_location'])
        if action_space == 'deliver':
            return 'deliver'
        elif action_space == 'pick':
            return 'pick'
        state = json.dumps(state)
        if mode == 'train':
            for action in action_space:
                if (state, action) not in self.q:
                    return action
            return random.choice(action_space)
        # print(action_space)
        # print(self.q)
        # p = random.random()
        # if p < self.epsilon:
        #     action = random.choice(action_space)
        #     # print(action)
        # else:
        q = [self.get_q(state, a) for a in action_space]
        # for act, score in zip(action_space, q):
        #     print(f'{act}: {score}')
        # print('-' * 30)
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
        :return: a list of possible actions where each action is represented by a tuple
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

    def print_q(self):
        for k, v in self.q.items():
            print(k, v)
        print('-' * 30)

    def learn(self, state1, action1, reward, state2):
        """
        Q-learning:
            Q(s, a) += alpha * (reward_func(s,a) + max(Q(s')) - Q(s,a))
        """
        # print(action1)
        # self.print_q()
        # REDUCE_FACTOR = 0.5
        penalty, bonus = 0, 0
        # x, y = state2['drone_location']
        # packages = {pack[0]: pack[1] for pack in state2['packages'] if type(pack[1]) == tuple}
        # drone_packages = [val[1] for val in state2['packages'] if type(val[1]) != tuple]

        # if action1 in {'deliver', 'pick'}:
        #     # after each pick or deliver action we modify the epsilon parameter
        #     # lowering the epsilon means we have more confident in our next moves
        #     self.epsilon = max(0.0, self.epsilon - 0.05)

        if action1 == 'pick' and reward > 0:
            reward = 40

        # if len(drone_packages) == 2:
        #     # the drone has 2 packages on it, we should only care about target location
        #     penalty += manhattan((x, y), state1['target_location'])
        #     # penalty += manhattan((x, y), state1['target_location'])
        #
        # else:
        #     dist_to_target = math.inf
        #     if drone_packages:
        #         # exactly 1 package on drone
        #         # calculates the distance to target location
        #         dist_to_target = manhattan((x, y), state1['target_location'])
        #         # dist_to_target += manhattan((x, y), state1['target_location'])
        #
        #     closest_pack_dist = math.inf
        #     if packages:
        #         closest_pack_dist = self.get_closest_package(x, y, packages)
        #     if not drone_packages and not packages:
        #         penalty = 0
        #     else:
        #         penalty += min(dist_to_target, closest_pack_dist)
        # if action1 in {'deliver', 'pick'}:
        #     reward = reward + bonus
        # else:
        #     reward = reward + bonus - penalty

        state1 = json.dumps(state1)
        state2 = json.dumps(state2)
        q_max = max([self.get_q(state2, a) for a in self.actions])
        old_q = self.q.get((state1, action1), None)
        if old_q is None:
            self.q[(state1, action1)] = reward
        else:
            self.q[(state1, action1)] = old_q + self.alpha * (reward + self.gamma * q_max - old_q)

    def get_closest_package(self, x, y, packages):
        res = []
        for pack_loc in packages.values():
            dist = manhattan((x, y), pack_loc)
            # dist += manhattan((x, y), pack_loc)
            res.append(dist)
        return min(res)
