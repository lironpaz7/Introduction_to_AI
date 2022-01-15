import math

ids = ["311280283", "313535379"]

UNKNOWN = '_'


def euclidean(a, b):
    return sum([(x - y) ** 2 for x, y in zip(a, b)]) ** 0.5


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


class DroneAgent:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.mode = 'train'  # do not change this!
        self.map = [[UNKNOWN for j in range(m)] for i in range(n)]
        self.p = [[0 for j in range(m)] for i in range(n)]

    def print_map(self, current_loc):
        map_data = self.reset_map()
        x, y = current_loc
        map_data[x][y] = 'V'
        for row in map_data:
            print(row)

    def reset_map(self):
        return [[UNKNOWN for j in range(self.m)] for i in range(self.n)]

    def select_action(self, obs0):
        # print(self.p)
        # print_board(self.map)
        return self.select_best_action(obs0)

    def calc_bonus(self, map_element, x_tar, y_tar, x, y):
        BONUS_SCORE = 1
        PENALTY_SCORE = 2
        bonus, penalty = 0, 0
        # check relative position of target location
        if x_tar < x:
            # target is in some upper row
            if map_element == 'P_WU':
                bonus += BONUS_SCORE * self.p[x][y]
            elif map_element == 'P_WD':
                penalty += PENALTY_SCORE * self.p[x][y]

        elif x_tar > x:
            if map_element == 'P_WD':
                bonus += BONUS_SCORE * self.p[x][y]
            elif map_element == 'P_WU':
                penalty += PENALTY_SCORE * self.p[x][y]

        if y_tar < y:
            if map_element == 'P_WL':
                bonus += BONUS_SCORE * self.p[x][y]
            elif map_element == 'P_WR':
                penalty += PENALTY_SCORE * self.p[x][y]

        elif y_tar > y:
            if map_element == 'P_WR':
                bonus += BONUS_SCORE * self.p[x][y]
            elif map_element == 'P_WL':
                penalty += PENALTY_SCORE * self.p[x][y]

        return bonus, penalty

    def select_best_action(self, obs0):
        action_space = self.drones_action_builder(obs0)
        if not obs0['packages']:
            # print('----------------------------')
            return 'reset'
        elif 'deliver' in action_space:
            return 'deliver'
        elif 'pick' in action_space:
            return 'pick'
        # choose best move depend on packages and delivers
        # assign score for each move then pick the one that maximizes
        penalties = {}
        packages = {pack[0]: pack[1] for pack in obs0['packages'] if type(pack[1]) == tuple}
        drone_packages = [val[1] for val in obs0['packages'] if type(val[1]) != tuple]

        for action in action_space:
            penalty = 0
            x, y = obs0['drone_location']
            if action[0] == 'm':
                # it's a move action
                direction = action.split('_')[1]
                if direction == 'up':
                    x, y = x - 1, y
                    if self.map[x][y] == 'P_WD':
                        penalty += 1
                elif direction == 'down':
                    x, y = x + 1, y
                    if self.map[x][y] == 'P_WU':
                        penalty += 1
                elif direction == 'left':
                    x, y = x, y - 1
                    if self.map[x][y] == 'P_WR':
                        penalty += 1
                else:
                    x, y = x, y + 1
                    if self.map[x][y] == 'P_WL':
                        penalty += 1



            map_element = self.map[x][y]
            # investigate map element and probability
            x_tar, y_tar = obs0['target_location']
            bonus, neg_bonus = 0, 0

            if len(drone_packages) == 2:
                # the drone has 2 packages on it, we should only care about target location
                penalty += euclidean((x, y), obs0['target_location'])
                penalty += manhattan((x, y), obs0['target_location'])
                bonus, neg_bonus = self.calc_bonus(map_element, x_tar, y_tar, x, y)

            else:
                dist_to_target = math.inf
                if drone_packages:
                    # exactly 1 package on drone
                    # calculates the distance to target location
                    dist_to_target = euclidean((x, y), obs0['target_location'])
                    dist_to_target += manhattan((x, y), obs0['target_location'])
                    # bonus, neg_bonus = self.calc_bonus(map_element, x_tar, y_tar, x, y)
                closest_pack = math.inf
                if packages:
                    closest_pack = self.get_closest_package(map_element, x, y, packages)
                penalty += min(dist_to_target, closest_pack)
            penalties[action] = penalty - bonus + neg_bonus
        return min([(k, v) for k, v in penalties.items()], key=lambda a: a[1])[0]

    def get_closest_package(self, map_element, x, y, packages):
        res = []
        for pack_loc in packages.values():
            dist = euclidean((x, y), pack_loc)
            dist += manhattan((x, y), pack_loc)
            bonus, neg_bonus = 0, 0
            # if dist != 0:
            #     x_pack, y_pack = pack_loc
            #     bonus, neg_bonus = self.calc_bonus(map_element, x_pack, y_pack, x, y)
            res.append(dist + bonus - neg_bonus)
        return min(res)

    def drones_action_builder(self, obs):
        """
        Builds all possible actions from a given state
        :param state: game state
        :return: a list of possible actions where each action is represented by a tuple
        """

        action_space = set()
        x, y = obs['drone_location']
        packages = {pack[0]: pack[1] for pack in obs['packages'] if type(pack[1]) == tuple}
        drone_packages = [val[1] for val in obs['packages'] if type(val[1]) != tuple]
        target_loc = obs['target_location']

        # -------------------------- add moves and wait --------------------------

        # move up
        if x - 1 >= 0 and self.map[x - 1][y][0] != 'I':
            # legal position
            # add movement
            action_space.add('move_up')

        # move down
        if x + 1 < self.n and self.map[x + 1][y] != 'I':
            # legal position
            # add movement
            action_space.add('move_down')

        # move left
        if y - 1 >= 0 and self.map[x][y - 1] != 'I':
            # legal position
            # add movement
            action_space.add('move_left')

        # move right
        if y + 1 < self.m and self.map[x][y + 1] != 'I':
            # legal position
            # add movement
            action_space.add('move_right')

        action_space.add('wait')

        # -------------------------- add pick --------------------------
        for pack_coord in packages.values():
            if (x, y) == pack_coord:
                action_space.add('pick')

        # -------------------------- add deliver --------------------------

        if drone_packages and (x, y) == target_loc:
            action_space.add('deliver')

        # total_options = []
        # num_drones = 1
        # rows, cols = self.rows, self.cols
        # drones = utils.shuffled(state['drones'].items())
        # for drone_name, coordinate in drones:
        #     options = set()
        #     x, y = coordinate
        #     # pick up and deliver actions
        #     if self.is_package_exist(state['packages'], coordinate) and \
        #             self.drone_can_pick_up_package(state, drone_name):
        #         # there is a package
        #         # check if we can pick it up!!
        #         # add action
        #         package_names = self.get_packages_at_coordinate(state['packages'], x, y)
        #         for package_name in package_names:
        #             if num_drones == 1:
        #                 current_drone_list = self.get_drones_packages_list(state['drones_packages'], drone_name)
        #                 add = True
        #                 for d_name in state['drones'].keys():
        #                     if d_name != drone_name:
        #                         if state['drones'][d_name] == coordinate:
        #                             d_name_list = self.get_drones_packages_list(state['drones_packages'], d_name)
        #                             if len(d_name_list) < len(current_drone_list):
        #                                 add = False
        #                 if add:
        #                     options.add(('pick up', drone_name, package_name))
        #             else:
        #                 # check if other drone already has the pick up option for this package
        #                 found = False
        #                 for drone_num in range(num_drones - 1):
        #                     drone_option_set = total_options[drone_num]
        #                     option = ('pick up', drones[drone_num][0], package_name)
        #                     if option in drone_option_set:
        #                         found = True
        #                         break
        #                 if not found:
        #                     # check if there is another drone in my location that can pick up this package and
        #                     # has less packages on itself
        #                     current_drone_list = self.get_drones_packages_list(state['drones_packages'], drone_name)
        #                     add = True
        #                     for d_name in state['drones'].keys():
        #                         if d_name != drone_name:
        #                             if state['drones'][d_name] == coordinate:
        #                                 d_name_list = self.get_drones_packages_list(state['drones_packages'],
        #                                                                             d_name)
        #                                 if len(d_name_list) < len(current_drone_list):
        #                                     add = False
        #                     if add:
        #                         options.add(('pick up', drone_name, package_name))
        #
        #     for pack_name in self.get_drones_packages_list(state['drones_packages'], drone_name):
        #         # check all packages for this specific drone
        #         client_name = self.who_can_receive_package(state['clients'], pack_name, x, y)
        #         if client_name is not None:
        #             # there is a client that can receive the 'pack_name'
        #             # add action
        #             options.add(('deliver', drone_name, client_name, pack_name))
        #
        #     best_move = self.find_best_move2(drone_name, state, rows, cols)
        #     if best_move is not None:
        #         options.add(best_move)
        #     else:
        #         if len(options) == 0:
        #             options.add(('wait', drone_name))
        #
        #     total_options.append(options)
        #     num_drones += 1
        #
        # if len(total_options) == 1:
        #     return list(total_options[0])
        return action_space

    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def update(self, obs0, action, obs1, reward):
        prev_loc = obs0['drone_location']
        current_loc = obs1['drone_location']
        # self.print_map(current_loc)
        # print(f'prev: {prev_loc}, curr: {current_loc}')
        if action in {'deliver', 'pick'} and reward < 0:
            # then our action didn't succeed
            # we don't want to update the map at all
            return
        if prev_loc == current_loc:
            x, y = prev_loc
            # we are in the same location so we need to check what was our move in order to
            # update our knowledge of the current map state
            # action_space = ['reset', 'wait', 'pick', 'move_up', 'move_down', 'move_left', 'move_right', 'deliver']
            if action[0] == 'm':
                # last action was a move operation so we need to check why we couldn't move there
                direction = action.split('_')[1]
                if direction == 'up' and self.p[x - 1][y] != 1:
                    # we tried to move up
                    x, y = x - 1, y
                    self.map[x][y] = 'P_WD'
                    self.p[x][y] = 0.5
                elif direction == 'down' and self.p[x + 1][y] != 1:
                    # we tried to move down
                    x, y = x + 1, y
                    self.map[x][y] = 'P_WU'
                    self.p[x][y] = 0.5
                elif direction == 'left' and self.p[x][y - 1] != 1:
                    # we tried to move left
                    x, y = x, y - 1
                    self.map[x][y] = 'P_WR'
                    self.p[x][y] = 0.5
                elif direction == 'right' and self.p[x][y + 1] != 1:
                    # we tried to move right
                    x, y = x, y + 1
                    self.map[x][y] = 'P_WL'
                    self.p[x][y] = 0.5

        else:
            # we were able to move
            # we should check if we are not in the exact place we wanted to be
            # then there is a wind for sure
            x_curr, y_curr = prev_loc
            x_new, y_new = current_loc
            # we are in the same location so we need to check what was our move in order to
            # update our knowledge of the current map state
            # action_space = ['reset', 'wait', 'pick', 'move_up', 'move_down', 'move_left', 'move_right', 'deliver']
            if action[0] == 'm':
                # last action was a move operation so we need to check why we couldn't move there
                direction = action.split('_')[1]
                if direction == 'up':
                    x_curr, y_curr = x_curr - 1, y_curr
                    if x_new < x_curr and y_curr == y_new:
                        # there is a wind up in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WU'
                        self.p[x_curr][y_curr] = 1
                    elif abs(x_new - x_curr) == 2 and y_curr == y_new:
                        # there is a wind down in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WD'
                        self.p[x_curr][y_curr] = 1
                    elif y_new < y_curr and x_curr == x_new:
                        # there is a wind left in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WL'
                        self.p[x_curr][y_curr] = 1
                    elif y_new > y_curr and x_curr == x_new:
                        # there us a wind right in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WR'
                        self.p[x_curr][y_curr] = 1

                elif direction == 'down':
                    x_curr, y_curr = x_curr + 1, y_curr
                    if x_new > x_curr and y_curr == y_new:
                        # there is a wind down in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WD'
                        self.p[x_curr][y_curr] = 1
                    elif abs(x_new - x_curr) == 2 and y_curr == y_new:
                        # there us a wind up in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WU'
                        self.p[x_curr][y_curr] = 1
                    elif y_new < y_curr and x_curr == x_new:
                        # there us a wind left in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WL'
                        self.p[x_curr][y_curr] = 1
                    elif y_new > y_curr and x_curr == x_new:
                        # there us a wind right in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WR'
                        self.p[x_curr][y_curr] = 1

                elif direction == 'left':
                    x_curr, y_curr = x_curr, y_curr - 1
                    if x_new > x_curr and y_curr == y_new:
                        # there is a wind down in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WD'
                        self.p[x_curr][y_curr] = 1
                    elif x_new < x_curr and y_curr == y_new:
                        # there us a wind up in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WU'
                        self.p[x_curr][y_curr] = 1
                    elif y_new < y_curr and x_curr == x_new:
                        # there us a wind left in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WL'
                        self.p[x_curr][y_curr] = 1
                    elif abs(y_new - y_curr) == 2 and x_curr == x_new:
                        # there us a wind right in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WR'
                        self.p[x_curr][y_curr] = 1

                elif direction == 'right':
                    x_curr, y_curr = x_curr, y_curr + 1
                    if x_new > x_curr and y_curr == y_new:
                        # there is a wind down in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WD'
                        self.p[x_curr][y_curr] = 1
                    elif x_new < x_curr and y_curr == y_new:
                        # there us a wind up in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WU'
                        self.p[x_curr][y_curr] = 1
                    elif y_new > y_curr and x_curr == x_new:
                        # there us a wind right in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WR'
                        self.p[x_curr][y_curr] = 1
                    elif abs(y_new - y_curr) == 2 and x_curr == x_new:
                        # there us a wind left in (x_curr, y_curr)
                        self.map[x_curr][y_curr] = 'P_WL'
                        self.p[x_curr][y_curr] = 1
            elif action[0] != 'r':
                # we should have stayed in the same location.
                # if we didn't there is a wind!!!!!!!!!!!!!!!
                if x_new > x_curr and y_curr == y_new:
                    # there is a wind down in (x_curr, y_curr)
                    self.map[x_curr][y_curr] = 'P_WD'
                    self.p[x_curr][y_curr] = 1
                elif x_new < x_curr and y_curr == y_new:
                    # there us a wind up in (x_curr, y_curr)
                    self.map[x_curr][y_curr] = 'P_WU'
                    self.p[x_curr][y_curr] = 1
                elif y_new > y_curr and x_curr == x_new:
                    # there us a wind right in (x_curr, y_curr)
                    self.map[x_curr][y_curr] = 'P_WR'
                    self.p[x_curr][y_curr] = 1
                elif y_new < y_curr and x_curr == x_new:
                    # there us a wind left in (x_curr, y_curr)
                    self.map[x_curr][y_curr] = 'P_WL'
                    self.p[x_curr][y_curr] = 1
