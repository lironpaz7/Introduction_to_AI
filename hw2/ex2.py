import random
import time

from hw2 import utils

ids = ["311280283", "313535379"]
import itertools
import networkx as nx

PACKAGE_ON_DRONE_WEIGHT = 2.4
PACKAGE_ON_BOARD_WEIGHT = 7.2
DISTANCE_TO_CLIENTS_WEIGHT = 1.2


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def euclidean(a, b):
    return sum([(x - y) ** 2 for x, y in zip(a, b)]) ** 0.5


class DroneAgent:
    def __init__(self, initial):
        # updates the map
        G = self._build_graph(initial['map'])
        self._short_distances = self._create_shortest_path_distances(G)
        self.initial_turns_to_go = initial['turns to go']
        self.num_initial_relevant_packages = 0
        self.relevant_packages = set()
        for client_name, client_data in initial['clients'].items():
            self.relevant_packages.update(set(client_data['packages']))
            self.num_initial_relevant_packages += len(client_data['packages'])
        self.turns_to_deliver_2_packages = []
        self.turns = 0
        self.delivers = 0

    def act(self, state):
        actions = self.drones_action_builder(state)
        num_drones = len((state["drones"]))
        return self._strategy(state, actions, num_drones)

    def get_num_packages_on_drones(self, packages):
        num = 0
        for v in packages.values():
            if type(v) != tuple:
                num += 1
        return num

    def _strategy(self, state, actions, num_drones):
        rows, cols = len(state['map']), len(state['map'][0])
        packages_on_board = len(
            {pack_name for pack_name in state['packages'].keys()}.intersection(self.relevant_packages))
        packages_on_drones = self.get_num_packages_on_drones(state['packages'])
        if self.delivers >= 2:
            self.turns_to_deliver_2_packages.append(self.turns)
            self.delivers = 0
            self.turns = 0

        if packages_on_board == 0 and packages_on_drones == 0:
            if self.num_initial_relevant_packages < 2:
                return 'terminate'
            else:
                # how much turns in average it takes to deliver 2 packages
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
                x, y = list(state['drones'].values())[0]
                if action[0] == 'move':
                    x, y = action[2]
                if action[0] == 'deliver':
                    score += 1000
                if action[0] == 'pick up':
                    score += 100
                if packages_on_drones > 0:
                    # distances to client
                    drone_name = action[1]
                    for package_name in self.get_drones_packages_list(state['packages'], drone_name):
                        client_name = self.get_client_from_package_name(state['clients'], package_name)
                        # check the distance to the client - in the next turn!
                        prob, loc = state['clients'][client_name]['probabilities'], state['clients'][client_name][
                            'location']
                        client_loc_next_turn = self.geuss_client_loc(prob, loc, rows, cols)
                        distances_to_clients += euclidean((x, y), tuple(client_loc_next_turn))
                        # key = (tuple(drone_loc), tuple(client_loc_next_turn))
                        # if key in self._short_distances:
                        #     distances_to_clients += self._short_distances[key]

                # distances to packages
                for package_loc in state['packages'].values():
                    # manhattan distances
                    key = ((x, y), tuple(package_loc))
                    if key in self._short_distances:
                        distances_to_packages += self._short_distances[key]
                        if distances_to_packages == 0:
                            # bonus for intersection with package
                            distances_to_packages -= 15

                score = score - distances_to_packages - packages_on_drones * PACKAGE_ON_DRONE_WEIGHT - \
                        packages_on_board * PACKAGE_ON_BOARD_WEIGHT - \
                        distances_to_clients * DISTANCE_TO_CLIENTS_WEIGHT
                scores_dict[action] = score
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
                    if len(self.get_drones_packages_list(state['packages'], drone_name)) > 0:
                        # distances to client
                        for package_name in self.get_drones_packages_list(state['packages'], drone_name):
                            client_name = self.get_client_from_package_name(state['clients'], package_name)
                            # check the distance to the client - in the next turn!
                            prob, loc = state['clients'][client_name]['probabilities'], \
                                        state['clients'][client_name][
                                            'location']
                            client_loc_next_turn = self.geuss_client_loc(prob, loc, rows, cols)
                            distances_to_clients += euclidean((x, y), tuple(client_loc_next_turn))
                            # key = (tuple(drone_loc), tuple(client_loc_next_turn))
                            # if key in self._short_distances:
                            #     distances_to_clients += self._short_distances[key]

                    # distances to packages
                    for package_loc in state['packages'].values():
                        # manhattan distances
                        key = ((x, y), tuple(package_loc))
                        if key in self._short_distances:
                            distances_to_packages += self._short_distances[key]
                            if distances_to_packages == 0:
                                # bonus for intersection with package
                                distances_to_packages -= 15

                    score = score - distances_to_packages - packages_on_drones * PACKAGE_ON_DRONE_WEIGHT - \
                            packages_on_board * PACKAGE_ON_BOARD_WEIGHT - \
                            distances_to_clients * DISTANCE_TO_CLIENTS_WEIGHT
                    total_score += score
                scores_dict[action] = total_score
                if best_score is None or score > best_score:
                    best_score, best_action = score, action

        # print(scores_dict)
        self.turns += 1
        if num_drones == 1:
            if best_action[0] == 'deliver':
                self.delivers += 1
            return [best_action]
        for act in best_action:
            if act[0] == 'deliver':
                self.delivers += 1
        return best_action

    def _h1_copy3(self, node):
        """
        Best h so far
        :param node:
        :return:
        """
        state = json.loads(node.state)
        circles = 0
        if node.action is not None:
            if type(node.action[0]) != tuple:
                if node.action[0] == 'deliver':
                    # deliver is the best choice always!!! we are not stupid
                    return -10
                elif node.action[0] == 'move':
                    # check if we are in a cycle --> penalty of 10
                    parent = node.parent
                    if parent is not None:
                        parent = parent.parent
                        if parent is not None:
                            parent_state = json.loads(parent.state)
                            for drone_name, drone_loc in parent_state['drones'].items():
                                current_turn_drone_name, current_turn_drone_move = drone_name, tuple(
                                    state['drones'][drone_name])
                                if tuple(drone_loc) == current_turn_drone_move:
                                    circles += 10
            else:
                for act in node.action:
                    if act[0] == 'deliver':
                        return -10
                    elif node.action[0] == 'move':
                        # check if we are in a cycle --> penalty of 10
                        parent = node.parent
                        if parent is not None:
                            parent = parent.parent
                            if parent is not None:
                                parent_state = json.loads(parent.state)
                                current_turn_drone_name = node.action[1]
                                current_turn_drone_move = tuple(state['drones'][current_turn_drone_name])
                                parent_drone_loc = tuple(parent_state['drones'][current_turn_drone_name])
                                if parent_drone_loc == current_turn_drone_move:
                                    circles += 10

        packages_on_board = len(state['packages'].keys())
        packages_on_drones = sum([len(p) for p in state['drones_packages']])
        distances_manhattan, distances_to_clients = 0, 0

        if packages_on_drones > 0:
            # dist to client
            for drone_name, drone_loc in state['drones'].items():
                for package_name in state['drones_packages'][drone_name]:
                    client_name = self.get_client_from_package_name(state['clients'], package_name)
                    # check the distance to the client - in the next turn!
                    client_loc_next_turn = self.geuss_client_loc()
                    distances_to_clients += manhattan(tuple(drone_loc), tuple(client_loc_next_turn))
                    # key = (tuple(drone_loc), tuple(client_loc_next_turn))
                    # if key in self._short_distances:
                    #     distances_to_clients += self._short_distances[key]

        # distances to packages
        for drone_loc in state['drones'].values():
            for package_loc in state['packages'].values():
                # manhattan distances
                key = (tuple(drone_loc), tuple(package_loc))
                if key in self._short_distances:
                    distances_manhattan += self._short_distances[key]
                if distances_manhattan == 0:
                    # bonus for intersection with client
                    distances_manhattan -= 1

        penalty = distances_manhattan + (packages_on_drones * 3.5) + (
                packages_on_board * 7.2) + distances_to_clients + circles + node.path_cost
        # print(penalty)
        return penalty

    def geuss_client_loc(self, prob, loc, rows, cols):
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

    def drones_action_builder(self, state):
        """
        Builds all possible actions from a given state
        :param state: game state
        :return: a tuple of possible actions where each action is represented by a tuple
        """
        total_options = []
        num_drones = 1
        rows, cols = len(state['map']), len(state['map'][0])
        drones = utils.shuffled(state['drones'].items())
        for drone_name, coordinate in drones:
            options = set()
            x, y = coordinate
            # pick up and deliver actions
            if state['map'][x][y] != 'I' and self.is_package_exist(state['packages'], coordinate):
                # there is a package
                # check if we can pick it up!!
                if self.drone_can_pick_up_package(state['packages'], drone_name):
                    # we can pick it up
                    # extract package name
                    # add action
                    package_names = self.get_packages_at_coordinate(state['packages'], x, y)
                    for package_name in package_names:
                        if num_drones == 1:
                            current_drone_list = self.get_drones_packages_list(state['packages'], drone_name)
                            add = True
                            for d_name in state['drones'].keys():
                                if d_name != drone_name:
                                    if state['drones'][d_name] == coordinate:
                                        d_name_list = self.get_drones_packages_list(state['packages'], d_name)
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
                                current_drone_list = self.get_drones_packages_list(state['packages'], drone_name)
                                add = True
                                for d_name in state['drones'].keys():
                                    if d_name != drone_name:
                                        if state['drones'][d_name] == coordinate:
                                            d_name_list = self.get_drones_packages_list(state['packages'], d_name)
                                            if len(d_name_list) < len(current_drone_list):
                                                add = False
                                if add:
                                    options.add(('pick up', drone_name, package_name))

            for pack_name in self.get_drones_packages_list(state['packages'], drone_name):
                # check all packages for this specific drone
                client_name = self.who_can_receive_package(state['clients'], pack_name, x, y)
                if client_name is not None:
                    # there is a client that can receive the 'pack_name'
                    # add action
                    options.add(('deliver', drone_name, client_name, pack_name))

            # add moves

            # move up
            if x - 1 >= 0 and state['map'][x - 1][y] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x - 1, y)))

            # move up-right
            if x - 1 >= 0 and y + 1 < cols and state['map'][x - 1][y + 1] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x - 1, y + 1)))

            # move up-left
            if x - 1 >= 0 and y - 1 >= 0 and state['map'][x - 1][y - 1] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x - 1, y - 1)))

            # move down
            if x + 1 < rows and state['map'][x + 1][y] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x + 1, y)))

            # move down-left
            if x + 1 < rows and y - 1 >= 0 and state['map'][x + 1][y - 1] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x + 1, y - 1)))

            # move down-right
            if x + 1 < rows and y + 1 < cols and state['map'][x + 1][y + 1] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x + 1, y + 1)))

            # move left
            if y - 1 >= 0 and state['map'][x][y - 1] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x, y - 1)))

            # move right
            if y + 1 < cols and state['map'][x][y + 1] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x, y + 1)))

            # add wait action as it is always possible
            options.add(('wait', drone_name))
            total_options.append(options)
            num_drones += 1

        if len(total_options) == 1:
            return list(total_options[0])
            # for act in total_options[0]:
            #     yield act
        else:
            # combinations of options
            return list(itertools.product(*total_options))
            # for act in total_options:
            #     yield act

    def create_packages_counter(self, clients):
        """
        Creates a dictionary with keys as clients and values as number of wanted packages
        :return: dictionary
        """
        d = {}
        for client in clients.keys():
            d[client] = len(clients[client]['packages'])
        return d

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
        creates shortest paths dictrionary
        :param G: graph object
        :return: dictrionary of shortest paths
        """
        d = {}
        for n1 in G.nodes:
            for n2 in G.nodes:
                if n1 == n2:
                    d[(n1, n2)] = 0
                else:
                    d[(n1, n2)] = len(nx.shortest_path(G, n1, n2)) - 1
        return d

    def _create_package_name_to_client_name(self, clients):
        """
        Creates a dictionary where package_name is a key and client name is value
        :return: dictionary
        """
        res = {}
        for name, d in clients.items():
            packages = d['packages']
            for package in packages:
                res[package] = name
        return res

    def _update_map(self, game_map, packages):
        """
        Updates the map with the given packages, increment the amount of packages in the specific cell.
        """
        for x, y in packages.values():
            if game_map[x][y] == 'I':
                continue
            if game_map[x][y] == 'P':
                game_map[x][y] = 1
            else:
                game_map[x][y] += 1

    def get_drones_packages_list(self, packages, drone_name):
        """
        Creates a list of package names that are on a specific drone_name
        :param packages: game packages
        :param drone_name: the name of the drone
        :return: list
        """
        return [pack_name for pack_name, coordinate in packages.items() if coordinate == drone_name]

    def drone_can_pick_up_package(self, packages, drone_name):
        """
        Checks if the specific drone can pick up another package
        :param drone_name: specific drone
        :return: True if possible, False otherwise
        """
        num = 0
        for v in packages.values():
            if v == drone_name:
                num += 1
        return num < 2

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
        :param x: pos x
        :param y: pos y
        :param package_name: package name
        :return: client name or None if doesn't exist
        """
        for name, client_data in clients.items():
            if package_name in client_data['packages']:
                return name
        return None
