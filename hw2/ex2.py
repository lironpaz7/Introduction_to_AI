import random
import time

from hw2 import utils

ids = ["311280283", "313535379"]
import itertools
import networkx as nx

PACKAGE_ON_DRONE_WEIGHT = 3.5
PACKAGE_ON_BOARD_WEIGHT = 7.2
DISTANCE_TO_CLIENTS_WEIGHT = 1


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def euclidean(a, b):
    return sum([(x - y) ** 2 for x, y in zip(a, b)]) ** 0.5


class DroneAgent:
    def __init__(self, initial):
        # updates the map
        G = self._build_graph(initial['map'])
        self._short_distances, self._short_distances_paths = self._create_shortest_path_distances(G)
        self.initial_turns_to_go = initial['turns to go']

        self.rows, self.cols = len(initial['map']), len(initial['map'][0])

        initial['drones_packages'] = {drone_name: [] for drone_name in initial['drones'].keys()}

        relevant_packages = set()
        for client in initial['clients'].values():
            for package in client['packages']:
                relevant_packages.add(package)
        # updates packages with relevant packages only
        for package in set(initial['packages'].keys()).difference(relevant_packages):
            initial['packages'].pop(package)

        self.num_initial_relevant_packages = len(initial['packages'].keys())
        self.turns_to_deliver_2_packages = []
        self.turns = 0
        self.delivers = 0

    def act(self, state):
        actions = self.drones_action_builder(state)
        num_drones = len((state["drones"]))
        best_action = self._strategy(state, actions, num_drones)
        if best_action == 'reset':
            self.delivers = 0
            self.turns = 0
        return best_action

    def get_num_packages_on_drones(self, drones_packages):
        return sum([len(drones) for drones in drones_packages.values()])

    def _strategy(self, state, actions, num_drones):
        rows, cols = self.rows, self.cols
        packages_on_drones = self.get_num_packages_on_drones(state['drones_packages'])
        packages_on_board = len(state['packages']) - packages_on_drones
        if self.delivers >= 2:
            self.turns_to_deliver_2_packages.append(self.turns)

        if packages_on_board == 0 and packages_on_drones == 0:
            if self.num_initial_relevant_packages < 2:
                return 'terminate'
            else:
                # how much turns in average it takes to deliver the first 2 packages
                # because we would like to reset only if we can deliver at least 2 packages
                average_turns_to_2_deliver = sum(self.turns_to_deliver_2_packages) / len(
                    self.turns_to_deliver_2_packages)
                if state['turns to go'] >= average_turns_to_2_deliver:
                    return 'reset'
                else:
                    return 'terminate'

        best_action, best_score = None, None
        scores_dict = {}

        for action in actions:
            score = 0
            if num_drones == 1:
                distances_to_packages, distances_to_clients = 0, 0
                drone_name = action[1]
                x, y = list(state['drones'].values())[0]
                if action[0] == 'move':
                    x, y = action[2]
                if action[0] == 'deliver':
                    score += 1000
                if action[0] == 'pick up':
                    score += 100
                if packages_on_drones > 0:
                    # distances to client
                    for package_name in state['drones_packages'][drone_name]:
                        client_name = self.get_client_from_package_name(state['clients'], package_name)
                        # check the distance to the client - in the next turn!
                        prob, loc = state['clients'][client_name]['probabilities'], state['clients'][client_name][
                            'location']
                        client_loc_next_turn = self.guess_client_loc(prob, loc, rows, cols)
                        key = ((x, y), tuple(client_loc_next_turn))
                        if key in self._short_distances:
                            distances_to_clients += self._short_distances[key]
                        else:
                            distances_to_clients += euclidean((x, y), tuple(client_loc_next_turn))

                # distances to packages
                # we will punish a drone:
                # 0 packages on it - 10 points
                # 1 package on it - 5 points
                # 2 package on it - 0 points
                num_packages_on_drone = len(state['drones_packages'][drone_name])
                if num_packages_on_drone == 0:
                    score -= 10
                if num_packages_on_drone == 1:
                    score -= 5
                if num_packages_on_drone < 2:
                    for package_loc in state['packages'].values():
                        # manhattan distances
                        key = ((x, y), tuple(package_loc))
                        if key in self._short_distances:
                            distances_to_packages += self._short_distances[key]
                            if distances_to_packages == 0:
                                # bonus for intersection with package
                                distances_to_packages -= 30
                else:
                    score += 10

                score = score - distances_to_packages - packages_on_drones * PACKAGE_ON_DRONE_WEIGHT - \
                        packages_on_board * PACKAGE_ON_BOARD_WEIGHT - \
                        distances_to_clients * DISTANCE_TO_CLIENTS_WEIGHT
                if best_score is None or score > best_score:
                    best_score, best_action = score, action
            else:
                total_score = 0
                for act in action:
                    distances_to_packages, distances_to_clients = 0, 0
                    drone_name = act[1]
                    x, y = state['drones'][drone_name]
                    if act[0] == 'move':
                        x, y = act[2]
                    if act[0] == 'deliver':
                        score += 1000
                    if act[0] == 'pick up':
                        score += 100
                    num_packages_on_drone = state['drones_packages'][drone_name]
                    if len(num_packages_on_drone) > 0:
                        # distances to client
                        for package_name in num_packages_on_drone:
                            client_name = self.get_client_from_package_name(state['clients'], package_name)
                            # check the distance to the client - in the next turn!
                            prob, loc = state['clients'][client_name]['probabilities'], \
                                        state['clients'][client_name][
                                            'location']
                            client_loc_next_turn = self.guess_client_loc(prob, loc, rows, cols)
                            key = ((x, y), tuple(client_loc_next_turn))
                            if key in self._short_distances:
                                distances_to_clients += self._short_distances[key]
                            else:
                                distances_to_clients += euclidean((x, y), tuple(client_loc_next_turn))

                    # distances to packages
                    num_packages_on_drone = len(state['drones_packages'][drone_name])
                    pack_factor = 100
                    if num_packages_on_drone < 2:
                        for package_loc in state['packages'].values():
                            # manhattan distances
                            key = ((x, y), tuple(package_loc))
                            if key in self._short_distances:
                                distances_to_packages += (
                                        self._short_distances[key] + pack_factor * (2.5 - num_packages_on_drone))
                                if distances_to_packages == 0:
                                    # bonus for intersection with package
                                    distances_to_packages -= 45
                    else:
                        score += 10

                    score = score - distances_to_packages - packages_on_drones * PACKAGE_ON_DRONE_WEIGHT - \
                            packages_on_board * PACKAGE_ON_BOARD_WEIGHT - \
                            distances_to_clients * DISTANCE_TO_CLIENTS_WEIGHT
                    total_score += score
                scores_dict[action] = total_score
                if best_score is None or total_score > best_score:
                    best_score, best_action = total_score, action

        # print(scores_dict)
        self.turns += 1
        if num_drones == 1:
            if best_action[0] == 'deliver':
                state['drones_packages'][best_action[1]].remove(best_action[3])
                self.delivers += 1
            elif best_action[0] == 'pick up':
                state['drones_packages'][best_action[1]].append(best_action[2])
            return [best_action]
        for act in best_action:
            if act[0] == 'deliver':
                state['drones_packages'][act[1]].remove(act[3])
                self.delivers += 1
            elif act[0] == 'pick up':
                state['drones_packages'][act[1]].append(act[2])
        return best_action

    def guess_client_loc(self, prob, loc, rows, cols):
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        for _ in range(1000):
            movement = random.choices(movements, weights=prob)[0]
            new_coordinates = (loc[0] + movement[0], loc[1] + movement[1])
            if new_coordinates[0] < 0 or new_coordinates[1] < 0 or new_coordinates[0] >= rows or new_coordinates[
                1] >= cols:
                continue
            break
        else:
            new_coordinates = (loc[0], loc[1])
        return new_coordinates

    def is_package_exist(self, packages, coordinate):
        """
        Checks if there is a package in the given coordinate
        :param packages: game package
        :param coordinate: (x,y) coordinate
        :return: True if exists and False otherwise
        """
        for pack_coordinate in packages.values():
            if pack_coordinate == coordinate:
                return True
        return False

    def find_best_move2(self, drone_name, state, rows, cols):
        x, y = state['drones'][drone_name]
        num_packages_on_drone = state['drones_packages'][drone_name]
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        if len(num_packages_on_drone) > 0:
            plan_b = []
            for package_name in num_packages_on_drone:
                client_name = self.get_client_from_package_name(state['clients'], package_name)
                # check the distance to the client - in the next turn!
                prob, loc = state['clients'][client_name]['probabilities'], \
                            state['clients'][client_name][
                                'location']
                client_loc_next_turn = self.guess_client_loc(prob, loc, rows, cols)
                key = (x, y), tuple(client_loc_next_turn)
                if key in self._short_distances_paths:
                    if not self._short_distances_paths[key]:
                        return 'wait', drone_name
                    return 'move', drone_name, self._short_distances_paths[key][0]
                best_loc, best_dist = None, None
                for i, j in movements:
                    if 0 <= x + i < rows and 0 <= y + j < cols and state['map'][x + i][y + j] != 'I':
                        next_loc = x + i, y + j
                        dist = euclidean(next_loc, tuple(client_loc_next_turn))
                        if best_dist is None or dist < best_dist:
                            best_loc, best_dist = next_loc, dist
                        plan_b.append((best_loc, best_dist))

            if len(num_packages_on_drone) == 2:
                return 'move', drone_name, min(plan_b, key=lambda a: a[1])[0]

        # dist to packages:
        if len(num_packages_on_drone) < 2:
            closest_pack, dist = None, None
            for package_loc in state['packages'].values():
                key = ((x, y), tuple(package_loc))
                if key in self._short_distances:
                    d = self._short_distances[key]
                    if d == 0:
                        return 'wait', drone_name
                    if dist is None or d < dist:
                        closest_pack, dist = self._short_distances_paths[key][0], d
            if closest_pack is not None:
                return 'move', drone_name, closest_pack
            else:
                for i, j in movements:
                    if 0 <= x + i < rows and 0 <= y + j < cols and state['map'][x + i][y + j] != 'I':
                        next_loc = x + i, y + j
                        return 'move', drone_name, next_loc

    def find_best_move(self, moves, state, rows, cols):
        best_move, best_score = None, None
        for move in moves:
            score = 0
            distances_to_clients, distances_to_packages = 0, 0
            drone_name = move[1]
            x, y = state['drones'][drone_name]
            if move[0] == 'move':
                x, y = move[2]
            num_packages_on_drone = state['drones_packages'][drone_name]
            if len(num_packages_on_drone) > 0:
                # distances to client
                for package_name in num_packages_on_drone:
                    client_name = self.get_client_from_package_name(state['clients'], package_name)
                    # check the distance to the client - in the next turn!
                    prob, loc = state['clients'][client_name]['probabilities'], \
                                state['clients'][client_name][
                                    'location']
                    client_loc_next_turn = self.guess_client_loc(prob, loc, rows, cols)
                    distances_to_clients += euclidean((x, y), tuple(client_loc_next_turn))

            # distances to packages
            num_packages_on_drone = len(state['drones_packages'][drone_name])
            pack_factor = 100
            if num_packages_on_drone < 2:
                for package_loc in state['packages'].values():
                    # manhattan distances
                    key = ((x, y), tuple(package_loc))
                    if key in self._short_distances:
                        distances_to_packages += (
                                self._short_distances[key] + pack_factor * (1 - num_packages_on_drone))
                        if distances_to_packages == 0:
                            # bonus for intersection with package
                            distances_to_packages -= 15
            else:
                score += 10
            score = score - distances_to_packages - distances_to_clients * (3 - num_packages_on_drone)
            if best_score is None or score > best_score:
                best_move, best_score = move, score
        return best_move

    def drones_action_builder(self, state):
        """
        Builds all possible actions from a given state
        :param state: game state
        :return: a tuple of possible actions where each action is represented by a tuple
        """
        total_options = []
        num_drones = 1
        rows, cols = self.rows, self.cols
        drones = utils.shuffled(state['drones'].items())
        for drone_name, coordinate in drones:
            options = set()
            x, y = coordinate
            # pick up and deliver actions
            if self.is_package_exist(state['packages'], coordinate) and \
                    self.drone_can_pick_up_package(state, drone_name):
                # there is a package
                # check if we can pick it up!!
                # add action
                package_names = self.get_packages_at_coordinate(state['packages'], x, y)
                for package_name in package_names:
                    if num_drones == 1:
                        current_drone_list = self.get_drones_packages_list(state['drones_packages'], drone_name)
                        add = True
                        for d_name in state['drones'].keys():
                            if d_name != drone_name:
                                if state['drones'][d_name] == coordinate:
                                    d_name_list = self.get_drones_packages_list(state['drones_packages'], d_name)
                                    if len(d_name_list) < len(current_drone_list):
                                        add = False
                        if add:
                            options.add(('pick up', drone_name, package_name))
                    else:
                        # check if other drone already has the pick up option for this package
                        found = False
                        for drone_num in range(num_drones - 1):
                            drone_option_set = total_options[drone_num]
                            option = ('pick up', drones[drone_num][0], package_name)
                            if option in drone_option_set:
                                found = True
                                break
                        if not found:
                            # check if there is another drone in my location that can pick up this package and
                            # has less packages on itself
                            current_drone_list = self.get_drones_packages_list(state['drones_packages'], drone_name)
                            add = True
                            for d_name in state['drones'].keys():
                                if d_name != drone_name:
                                    if state['drones'][d_name] == coordinate:
                                        d_name_list = self.get_drones_packages_list(state['drones_packages'],
                                                                                    d_name)
                                        if len(d_name_list) < len(current_drone_list):
                                            add = False
                            if add:
                                options.add(('pick up', drone_name, package_name))

            for pack_name in self.get_drones_packages_list(state['drones_packages'], drone_name):
                # check all packages for this specific drone
                client_name = self.who_can_receive_package(state['clients'], pack_name, x, y)
                if client_name is not None:
                    # there is a client that can receive the 'pack_name'
                    # add action
                    options.add(('deliver', drone_name, client_name, pack_name))

            # -------------------------- add moves --------------------------
            # find the best move possible for this drone (all are independent of each other)
            # moves = set()
            # # move up
            # if x - 1 >= 0 and state['map'][x - 1][y] != 'I':
            #     # legal position
            #     # add movement
            #     moves.add(('move', drone_name, (x - 1, y)))
            #
            # # move up-right
            # if x - 1 >= 0 and y + 1 < cols and state['map'][x - 1][y + 1] != 'I':
            #     # legal position
            #     # add movement
            #     moves.add(('move', drone_name, (x - 1, y + 1)))
            #
            # # move up-left
            # if x - 1 >= 0 and y - 1 >= 0 and state['map'][x - 1][y - 1] != 'I':
            #     # legal position
            #     # add movement
            #     moves.add(('move', drone_name, (x - 1, y - 1)))
            #
            # # move down
            # if x + 1 < rows and state['map'][x + 1][y] != 'I':
            #     # legal position
            #     # add movement
            #     moves.add(('move', drone_name, (x + 1, y)))
            #
            # # move down-left
            # if x + 1 < rows and y - 1 >= 0 and state['map'][x + 1][y - 1] != 'I':
            #     # legal position
            #     # add movement
            #     moves.add(('move', drone_name, (x + 1, y - 1)))
            #
            # # move down-right
            # if x + 1 < rows and y + 1 < cols and state['map'][x + 1][y + 1] != 'I':
            #     # legal position
            #     # add movement
            #     moves.add(('move', drone_name, (x + 1, y + 1)))
            #
            # # move left
            # if y - 1 >= 0 and state['map'][x][y - 1] != 'I':
            #     # legal position
            #     # add movement
            #     moves.add(('move', drone_name, (x, y - 1)))
            #
            # # move right
            # if y + 1 < cols and state['map'][x][y + 1] != 'I':
            #     # legal position
            #     # add movement
            #     moves.add(('move', drone_name, (x, y + 1)))
            #
            # moves.add(('wait', drone_name))

            best_move = self.find_best_move2(drone_name, state, rows, cols)
            options.add(best_move)

            total_options.append(options)
            num_drones += 1

        if len(total_options) == 1:
            return list(total_options[0])
        else:
            # combinations of options
            return list(itertools.product(*total_options))

    def get_packages_at_coordinate(self, packages, x, y):
        """
        Creates a list of package names at coordinate (x,y)
        :param packages: game packages
        :param x: x coordinate
        :param y: y coordinate
        :return: list of packages names
        """
        return [pack_name for pack_name, pack_coordinate in packages.items() if (x, y) == pack_coordinate]

    def _build_graph(self, game_map):
        """
        Builds the game graph where a node is represented as (x,y) coordinate, and 'I' is not reachable node
        :param game_map: the map of the game
        :return: graph representing the game map
        """
        G = nx.Graph()
        rows, cols = len(game_map), len(game_map[0])
        for i in range(rows):
            for j in range(cols):
                if game_map[i][j] == 'I':
                    continue
                # edge from (i,j) to its adjacent: (i+1,j), (i-1,j), (i,j+1), (i,j-1)
                if i + 1 < rows and game_map[i + 1][j] != 'I':
                    G.add_edge((i, j), (i + 1, j))
                if i - 1 >= 0 and game_map[i - 1][j] != 'I':
                    G.add_edge((i, j), (i - 1, j))
                if j + 1 < cols and game_map[i][j + 1] != 'I':
                    G.add_edge((i, j), (i, j + 1))
                if j - 1 >= 0 and game_map[i][j - 1] != 'I':
                    G.add_edge((i, j), (i, j - 1))

                # edge from (i,j) to its adjacent diagonal: (i+1,j+1), (i+1,j-1), (i-1,j+1), (i-1,j-1)
                if i + 1 < rows and j + 1 < cols and game_map[i + 1][j + 1] != 'I':
                    G.add_edge((i, j), (i + 1, j + 1))
                if i + 1 < rows and j - 1 >= 0 and game_map[i + 1][j - 1] != 'I':
                    G.add_edge((i, j), (i + 1, j - 1))
                if i - 1 >= 0 and j + 1 < cols and game_map[i - 1][j + 1] != 'I':
                    G.add_edge((i, j), (i - 1, j + 1))
                if i - 1 >= 0 and j - 1 >= 0 and game_map[i - 1][j - 1] != 'I':
                    G.add_edge((i, j), (i - 1, j - 1))
        return G

    def _create_shortest_path_distances(self, G):
        """
        Creates shortest paths dictrionary
        :param G: graph object
        :return: dictrionary of shortest paths
        """
        paths_len, paths = {}, {}
        for n1 in G.nodes:
            for n2 in G.nodes:
                if n1 == n2:
                    paths[(n1, n2)] = []
                    paths_len[(n1, n2)] = 0
                else:
                    path = nx.shortest_path(G, n1, n2)[1:]
                    paths[(n1, n2)] = path
                    paths_len[(n1, n2)] = len(path)
        return paths_len, paths

    def get_drones_packages_list(self, drones_packages, drone_name):
        """
        Creates a list of package names that are on a specific drone_name
        :param packages: game packages
        :param drone_name: the name of the drone
        :return: list
        """
        return drones_packages[drone_name]

    def drone_can_pick_up_package(self, state, drone_name):
        """
        Checks if the specific drone can pick up another package
        :param drone_name: specific drone
        :return: True if possible, False otherwise
        """
        return len(state['drones_packages'][drone_name]) < 2

    def who_can_receive_package(self, clients, package_name, x, y):
        """
        Checks if there is a client at position (x,y) that would like to receive 'pack_name'
        :param x: pos x
        :param y: pos y
        :param package_name: package name
        :return: client name or None if doesn't exist
        """
        for name, client_data in clients.items():
            if package_name in client_data['packages']:
                client_pos = clients[name]['location']
                if (x, y) == client_pos:
                    return name
                else:
                    break
        return None

    def get_client_from_package_name(self, clients, package_name):
        """
        Checks if there is a client at position (x,y) that would like to receive 'pack_name'
        :param clients: dictionary of clients
        :param package_name: package name
        :return: client name or None if doesn't exist
        """
        for name, client_data in clients.items():
            if package_name in client_data['packages']:
                return name
        return None
