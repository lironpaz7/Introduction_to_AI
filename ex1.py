import itertools
import uuid

import search
import random
import copy
import json
import math
import networkx as nx

import utils

ids = ["311280283", "313535379"]


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def euclidean(a, b):
    return sum([(x - y) ** 2 for x, y in zip(a, b)]) ** 0.5


class DroneProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        game_map = initial['map']
        drones = initial['drones']
        packages = initial['packages']
        # updates the map
        self._update_map(game_map, packages)
        G = self._build_graph(game_map)
        self._short_distances = self._create_shortest_path_distances(G)
        clients = initial['clients']
        # goal
        client_packages_counter = self.create_packages_counter(clients)
        turns = 0
        # checks if there is no solution
        # self._are_all_packages_reachable(clients, game_map, packages)
        # creates the coordinate to package dictionary with relevant packages only
        relevant_packages = set()
        for client in clients.values():
            for package in client['packages']:
                relevant_packages.add(package)
        # updates packages with relevant packages only
        for package in set(packages.keys()).difference(relevant_packages):
            packages.pop(package)
        coordinate_to_packages = self.get_packages_at_coordinate(packages, relevant_packages)
        # -------- drones data --------
        # drones packages
        drones_packages = {name: [] for name in drones.keys()}
        self._package_name_to_client_name = self._create_package_name_to_client_name(clients)
        # set state as a tuple of wanted features
        state = {'clients': clients,
                 'game_map': game_map,
                 'drones': drones,
                 'packages': packages,
                 'coordinate_to_packages': coordinate_to_packages,
                 'client_packages_counter': client_packages_counter,
                 'drones_packages': drones_packages,
                 'turns': turns}
        state = json.dumps(state)
        search.Problem.__init__(self, state)

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        return self.drones_action_builder(state)

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        state = copy.deepcopy(state)
        state = json.loads(state)
        # iterate over all actions
        # print(action)
        if type(action[0]) == tuple:
            for drone_action in utils.shuffled(action):
                # print(drone_action)
                self.apply_action(state, drone_action)
        else:
            self.apply_action(state, action)
        state['turns'] += 1
        return json.dumps(state)

    def apply_action(self, state, drone_action):
        # action = wait --> ('wait', drone_name)
        if drone_action[0] == 'wait':
            pass
        # action = movement --> ('move', drone_name, (x, y))
        elif drone_action[0] == 'move':
            drone_name = drone_action[1]
            # extract drones new coordinates
            x_new, y_new = drone_action[2]
            state['drones'][drone_name] = (x_new, y_new)
        # action = pickup --> ('pick up', drone_name, package_name)
        elif drone_action[0] == 'pick up':
            drone_name = drone_action[1]
            package_name = drone_action[2]
            # extract package coordinates
            x, y = state['packages'][package_name]
            # delete package from game packages
            state['packages'].pop(package_name)
            state['coordinate_to_packages'][f'({x}, {y})'].remove(package_name)
            # add package to drone packages list
            state['drones_packages'][drone_name].append(package_name)
            # updates game board
            state['game_map'][x][y] -= 1
        # action = deliver --> ('deliver', drone_name, client_name, package_name)
        else:
            drone_name = drone_action[1]
            client_name = drone_action[2]
            package_name = drone_action[3]
            # remove package from drone packages list
            state['drones_packages'][drone_name].remove(package_name)
            # decrease wanted packages by this client
            state['client_packages_counter'][client_name] -= 1
            # maybe update clients["client_name"]["packages"]?

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        state = json.loads(state)
        return sum([packages_left for packages_left in state['client_packages_counter'].values()]) == 0

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        # state = json.loads(node.state)
        # print(state['turns'], state['drones_packages'], state['drones'], node.action)
        # return self._h2(node)
        return self._h1_copy3(node)

    def _h1(self, node):
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
                                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                                d1_name, d1_move = node.action[1], node.action[2]
                                # print(d1_name, drone_name, d1_name == drone_name)
                                # print(d1_move, drone_loc)
                                if drone_name == d1_name and tuple(drone_loc) == d1_move:
                                    circles += 100
            else:
                for act in node.action:
                    if act[0] == 'deliver':
                        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                        return -10
                    elif node.action[0] == 'move':
                        # implement cycle check for multiple drones
                        pass

        packages_on_board = len(state['packages'].keys()) * 8
        packages_on_drones = sum([len(p) for p in state['drones_packages']]) * 3
        distances_manhattan, distances_euclidean, distances_to_I = 0, 0, 0

        # distances_to_clients = 0
        # num_packages_on_drones = 0
        # for packages_list in state['drones_packages'].values():
        #     num_packages_on_drones += len(packages_list)
        # if num_packages_on_drones > 0:
        #     # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #     # dist to client
        #     for drone_name, drone_loc in state['drones'].items():
        #         for package_name in state['drones_packages'][drone_name]:
        #             client_name = self.create_package_name_to_client_name(state['clients'])[package_name]
        #             path = state['clients'][client_name]['path']
        #             # min_dist_to_client_path = min([manhattan(drone_loc, path[i]) for i in range(len(path))])
        #             client_loc_next_turn = utils.turn_heading(path[0], state['turns'] + 1, path)
        #             distances_to_clients += manhattan(drone_loc, client_loc_next_turn)

        for drone_loc in state['drones'].values():
            for package_loc in state['packages'].values():
                # manhattan distances
                distances_manhattan += manhattan(drone_loc, package_loc)
                # euclidean distances
                distances_euclidean += euclidean(drone_loc, package_loc)
                # for I_loc in self._where_I:
                #     distances_to_I += manhattan(drone_loc, I_loc)
                if distances_manhattan == 0:
                    # bonus for intersection with client
                    distances_manhattan -= 1
                    distances_euclidean -= 1
        penalty = max(distances_manhattan,
                      distances_euclidean) + packages_on_drones + packages_on_board + circles + node.path_cost
        # print(penalty)
        return penalty

    def _h1_copy(self, node):
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
                                    circles += 100
            else:
                for act in node.action:
                    if act[0] == 'deliver':
                        return -10
                    elif node.action[0] == 'move':
                        # implement cycle check for multiple drones
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
                                    circles += 100

        packages_on_board = len(state['packages'].keys())
        packages_on_drones = sum([len(p) for p in state['drones_packages']])
        distances_manhattan, distances_euclidean, distances_to_I, distances_to_clients = 0, 0, 0, 0

        if packages_on_drones > 0:
            # dist to client
            for drone_name, drone_loc in state['drones'].items():
                for package_name in state['drones_packages'][drone_name]:
                    client_name = self.create_package_name_to_client_name(state['clients'])[package_name]
                    path = state['clients'][client_name]['path']
                    # min_dist_to_client_path = min([manhattan(drone_loc, path[i]) for i in range(len(path))])
                    client_loc_next_turn = utils.turn_heading(path[0], state['turns'] + 1, path)
                    distances_to_clients += manhattan(drone_loc, client_loc_next_turn)

        for drone_loc in state['drones'].values():
            for package_loc in state['packages'].values():
                # manhattan distances
                distances_manhattan += manhattan(drone_loc, package_loc)
                # euclidean distances
                distances_euclidean += euclidean(drone_loc, package_loc)
                # for I_loc in self._where_I:
                #     distances_to_I += manhattan(drone_loc, I_loc)
                if distances_manhattan == 0:
                    # bonus for intersection with client
                    distances_manhattan -= 1
                    distances_euclidean -= 1

        penalty = max(distances_manhattan,
                      distances_euclidean) + (packages_on_drones * 3) + (
                          packages_on_board * 8) + distances_to_clients + circles + node.path_cost
        # print(penalty)
        return penalty

    def _h1_copy2(self, node):
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
                                    circles += 100
            else:
                for act in node.action:
                    if act[0] == 'deliver':
                        return -10
                    elif node.action[0] == 'move':
                        # implement cycle check for multiple drones
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
                                    circles += 100

        packages_on_board = len(state['packages'].keys())
        packages_on_drones = sum([len(p) for p in state['drones_packages']])
        distances_manhattan, distances_euclidean, distances_to_I, distances_to_clients = 0, 0, 0, 0

        if packages_on_drones > 0:
            # dist to client
            for drone_name, drone_loc in state['drones'].items():
                for package_name in state['drones_packages'][drone_name]:
                    client_name = self.create_package_name_to_client_name(state['clients'])[package_name]
                    path = state['clients'][client_name]['path']
                    # min_dist_to_client_path = min([manhattan(drone_loc, path[i]) for i in range(len(path))])
                    client_loc_next_turn = utils.turn_heading(path[0], state['turns'] + 1, path)
                    distances_to_clients += manhattan(drone_loc, client_loc_next_turn)

        for drone_loc in state['drones'].values():
            for package_loc in state['packages'].values():
                # manhattan distances
                distances_manhattan += manhattan(drone_loc, package_loc)
                # euclidean distances
                distances_euclidean += euclidean(drone_loc, package_loc)
                if distances_manhattan == 0:
                    # bonus for intersection with client
                    distances_manhattan -= 1
                    distances_euclidean -= 1

        penalty = max(distances_euclidean, distances_manhattan) + (packages_on_drones * 3) + (
                packages_on_board * 8) + distances_to_clients + circles + node.path_cost
        # print(penalty)
        return penalty

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
                                    circles += 100
            else:
                for act in node.action:
                    if act[0] == 'deliver':
                        return -10
                    elif node.action[0] == 'move':
                        # implement cycle check for multiple drones
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
                                    circles += 100

        packages_on_board = len(state['packages'].keys())
        packages_on_drones = sum([len(p) for p in state['drones_packages']])
        distances_manhattan, distances_euclidean, distances_to_clients = 0, 0, 0

        if packages_on_drones > 0:
            # dist to client
            for drone_name, drone_loc in state['drones'].items():
                for package_name in state['drones_packages'][drone_name]:
                    client_name = self._package_name_to_client_name[package_name]
                    path = state['clients'][client_name]['path']
                    # min_dist_to_client_path = min([manhattan(drone_loc, path[i]) for i in range(len(path))])
                    client_loc_next_turn = utils.turn_heading(path[0], state['turns'] + 1, path)
                    distances_to_clients += manhattan(tuple(drone_loc), tuple(client_loc_next_turn))
                    # key = (tuple(drone_loc), tuple(client_loc_next_turn))
                    # if key in self._short_distances:
                    #     distances_to_clients += self._short_distances[key]

        for drone_loc in state['drones'].values():
            for package_loc in state['packages'].values():
                # manhattan distances
                key = (tuple(drone_loc), tuple(package_loc))
                if key in self._short_distances:
                    distances_manhattan += self._short_distances[key]
                # distances_euclidean += euclidean(drone_loc, package_loc)
                if distances_manhattan == 0:
                    # bonus for intersection with client
                    distances_manhattan -= 1

        penalty = distances_manhattan + (packages_on_drones * 3) + (
                packages_on_board * 8) + distances_to_clients + circles + node.path_cost
        # print(penalty)
        return penalty

    def _h3(self, node):
        state = json.loads(node.state)
        distances = 0
        # print(state['turns'], state['packages'], state['drones_packages'], state['drones'], node.action)
        if node.action is not None:
            if type(node.action[0]) != tuple:
                if node.action[0] == 'deliver':
                    # deliver is the best choice always!!! we are not stupid
                    return -10

            num_packages_on_drones = 0
            for packages_list in state['drones_packages'].values():
                num_packages_on_drones += len(packages_list)
            if num_packages_on_drones > 0:
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                # dist to client
                for drone_name, drone_loc in state['drones'].items():
                    for package_name in state['drones_packages'][drone_name]:
                        client_name = self.create_package_name_to_client_name(state['clients'])[package_name]
                        path = state['clients'][client_name]['path']
                        # min_dist_to_client_path = min([manhattan(drone_loc, path[i]) for i in range(len(path))])
                        client_loc_next_turn = utils.turn_heading(path[0], state['turns'] + 1, path)
                        distances += manhattan(drone_loc, client_loc_next_turn)
            else:
                # dist to packages
                for drone_loc in state['drones'].values():
                    for package_loc in state['packages'].values():
                        dist = manhattan(drone_loc, package_loc)
                        distances += dist
                        if dist == 0:
                            # bonus for intersection with client
                            distances -= 1
                distances += (len(state['packages'].keys()) * 5)
        # print(distances)
        return distances

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

    def drone_can_pick_up_package(self, drones_packages, drone_name):
        """
        Checks if the specific drone can pick up another package
        :param drone_name: specific drone
        :return: True if possible, False otherwise
        """
        return len(drones_packages[drone_name]) < 2

    def who_can_receive_package(self, clients, package_name, x, y, turns):
        """
        Checks if there is a client at position (x,y) that would like to receive 'pack_name'
        :param x: pos x
        :param y: pos y
        :param package_name: package name
        :return: client name or None if doesn't exist
        """
        client_name = self._package_name_to_client_name[package_name]
        client_movements = clients[client_name]['path']
        client_pos = utils.turn_heading(client_movements[0], turns, client_movements)
        return client_name if (x, y) == tuple(client_pos) else None

    def drones_action_builder(self, state):
        """
        Builds all possible actions from a given state
        :param state: game state
        :return: a tuple of possible actions where each action is represented by a tuple
        """
        state = json.loads(state)
        total_options = []
        num_drones = 1
        rows, cols = len(state['game_map']), len(state['game_map'][0])
        drones = utils.shuffled(state['drones'].items())
        for drone_name, coordinate in drones:
            options = set()
            x, y = coordinate
            # pick up and deliver actions
            if state['game_map'][x][y] not in {'P', 'I'} and state['game_map'][x][y] > 0:
                # there is a package
                # check if we can pick it up!!
                if self.drone_can_pick_up_package(state['drones_packages'], drone_name):
                    # we can pick it up
                    # extract package name
                    # add action
                    package_names = state['coordinate_to_packages'][f'({x}, {y})']
                    for package_name in package_names:
                        if num_drones == 1:
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
                                options.add(('pick up', drone_name, package_name))

            for pack_name in state['drones_packages'][drone_name]:
                # check all packages for this specific drone
                client_name = self.who_can_receive_package(state['clients'], pack_name, x, y, state['turns'])
                if client_name is not None:
                    # there is a client that can receive the 'pack_name'
                    # add action
                    options.add(('deliver', drone_name, client_name, pack_name))

            # move up
            if x - 1 >= 0 and state['game_map'][x - 1][y] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x - 1, y)))

                # move down
            if x + 1 < rows and state['game_map'][x + 1][y] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x + 1, y)))

            # move left
            if y - 1 >= 0 and state['game_map'][x][y - 1] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x, y - 1)))

            # move right
            if y + 1 < cols and state['game_map'][x][y + 1] != 'I':
                # legal position
                # add movement
                options.add(('move', drone_name, (x, y + 1)))

            # add wait action as it is always possible
            options.add(('wait', drone_name))
            total_options.append(tuple(options))
            num_drones += 1
        if len(total_options) == 1:
            for act in total_options[0]:
                yield act
        else:
            # combinations of options
            total_options = tuple(itertools.product(*total_options))
            for act in total_options:
                yield act

    # def _are_all_packages_reachable(self, clients, game_map, packages):
    #     """
    #     Checks if all the relevant packages are reachable
    #     :param map: game map
    #     :param packages: packages dictionary
    #     :return: True if reachable, False otherwise
    #     """
    #     for client, _ in clients.items():
    #         client_packages = clients[client]['packages']
    #         for package_name in client_packages:
    #             if package_name not in packages:
    #                 return False
    #             # checks if the package reachable
    #             x, y = packages[package_name]
    #             if game_map[x][y] == 'I':
    #                 return False
    #     return True

    def create_packages_counter(self, clients):
        """
        Creates a dictionary with keys as clients and values as number of wanted packages
        :return: dictionary
        """
        d = {}
        for client in clients.keys():
            d[client] = len(clients[client]['packages'])
        return d

    def get_packages_at_coordinate(self, packages, relevant_packages):
        """
        Creates a dictionary with keys as coordinates and values as package name
        :param relevant_packages: relevant game packages
        :param packages: game packages
        :return:
        """
        position_to_packages = {}
        for package_name, coordinate in packages.items():
            if package_name not in relevant_packages:
                continue
            coordinate = str(tuple(coordinate))
            if coordinate in position_to_packages:
                position_to_packages[coordinate].append(package_name)
            else:
                position_to_packages[coordinate] = [package_name]
        return position_to_packages

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
        return G

    def _create_shortest_path_distances(self, G):
        d = {}
        for n1 in G.nodes:
            for n2 in G.nodes:
                if n1 == n2:
                    continue
                d[(n1, n2)] = len(nx.shortest_path(G, n1, n2)) - 1
        return d


def create_drone_problem(game):
    return DroneProblem(game)
