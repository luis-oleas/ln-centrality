import os
from itertools import cycle
import logging
import numpy as np
import networkx as nx
import ln.utils as utils
import ln.data_analysis as data_analysis
from datetime import datetime

# Constants for describing lightning clients implementations
IMPL_C_LIGHTNING = "c-lightning"
IMPL_LND = "lnd"
IMPL_LND_0_6 = "lnd_0.6"
IMPL_ECLAIR = "eclair"
IMPL_INCONCLUSIVE = "inconclusive"
IMPL_COLLIDING = "colliding"
IMPL_NODE = "IMPLEMENTATION/NODE_NAME"

IMPLEMENTATION_PARAMS = {
    IMPL_C_LIGHTNING: {'time_lock_delta': 14, 'fee_base_msat': '1000', 'fee_rate_milli_msat': '10'},
    IMPL_LND: {'time_lock_delta': 144, 'fee_base_msat': '1000', 'fee_rate_milli_msat': '1'},
    IMPL_LND_0_6: {'time_lock_delta': 40, 'fee_base_msat': '1000', 'fee_rate_milli_msat': '1'},
    IMPL_ECLAIR: {'time_lock_delta': 144, 'fee_base_msat': '1000', 'fee_rate_milli_msat': '100'}
}


class LNCentrality:
    """
        Structure used as main body of the simulation, it handles either the initial configuration to connect to a node
        or loading of a snapshot
    """

    def __init__(self, json_filename_temp: str, implementation: dict = None, balance: dict = None, htlc: dict = None):
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))) + (
                            '\\data' if os.sys.platform == 'win32' else '/data')
        self.location_snapshots = self.location + ('\\snapshot' if os.sys.platform == 'win32' else '/snapshot')
        self.location_results = self.location + ('\\results' if os.sys.platform == 'win32' else '/results')

        self.name = json_filename_temp
        self.parameters = utils.load_file(self.location, "parameters.json", False)
        self.implementation = implementation
        self.balance = balance
        self.htlc = htlc
        self.parameters['location_results'] = self.location_results

        """
        Load data from json_filename and fill in all the data we know for the two graphs.
        Data for all fields except those starting with * can be found in the json file.
        g1:
            nodes
                pub_key (id)
                last_update
                alias
                addresses
                color
                features
                * implementation
            edges
                channel_id (id)
                chan_point
                capacity
                last_update
                node1_pub
                node2_pub
                node1_policy
                node2_policy
                           
        g2:
            nodes
                pub_key (id)

            edges
                channel_id | node1_pub (id) 
                    o bé 
                channel_id | node2_pub (id) 
                * balance (float)
                * pending_htlc (dict)
        """
        data = None
        is_simulations = False
        while True:
            try:
                if input('Load set of Snapshots? (y/n):') == 'y':
                    data = utils.load_set_files(self.location_snapshots, '*.json')
                else:
                    # Function to load data from a json file and set its initial values
                    if input('Run two simulations for one snapshot? (y/n):') == 'y':
                        is_simulations = True
                        data = utils.load_file(self.location_snapshots, self.parameters['simulations']['file_name'], True)
                    else:
                        data = (utils.load_file(self.location_snapshots, self.name, True))
            except FileNotFoundError as f:
                print("{} FILE NOT FOUND:{}-{}".format(utils.spaces, f.strerror, f.filename))
                break
            else:
                break

        if data is not None:
            for dat in data:
                logging.debug("Processing".format(dat))
                # Gets the aim values for the simulations, specifically the dictionaries for the node and edge
                self.parameters['filename_snapshot'] = dat[0]['filename_snapshot'] \
                    if isinstance(dat, list) else dat['filename_snapshot']
                print("{}{} {} {}{}".format(utils.asterisks, utils.asterisks,
                                            self.parameters['filename_snapshot'].upper(), utils.asterisks,
                                            utils.asterisks, ))
                self.g1, self.g2, dict_graphs, self.nodeDict, self.edgeDict, link_nodes, extra_nodes = \
                    utils.populate_graphs((dat if 'data' not in dat and not isinstance(dat, list) else
                                           dat[0]['data'] if isinstance(dat, list) else dat['data']))

                self.__infer_implementation(self.implementation)
                self.__assign_rand_balances(self.balance)
                self.__assign_rand_htlc(self.htlc)

                if is_simulations:
                    iteration = len(self.parameters['simulations']['sims'])
                    for val in reversed(self.parameters['simulations']['sims']):
                        self.parameters['flag_length_path'] = val['flag_length_path']
                        self.parameters['flag_channel_enabled'] = val['flag_channel_enabled']
                        self.parameters['flag_enough_balance'] = val['flag_enough_balance']
                        self.parameters['flag_minimum_htlc'] = val['flag_minimum_htlc']
                        self.parameters['flag_valid_timeout'] = val['flag_valid_timeout']
                        self.parameters['flag_payment_greater_delta'] = val['flag_payment_greater_delta']
                        ordinal = '_' + str(iteration) + {1: "st", 2: "nd", 3: "rd"}.\
                            get(iteration if (iteration < 20) else (iteration % 10), 'th')
                        utils.set_centrality(self.g1.copy(), self.g2.copy(), dict_graphs.copy(), self.parameters,
                                             link_nodes, extra_nodes, ordinal)
                        iteration -= 1
                else:
                    utils.set_centrality(self.g1.copy(), self.g2.copy(), dict_graphs, self.parameters, link_nodes,
                                         extra_nodes)

                print(nx.info(self.g1))
                print(nx.info(self.g2))

            print("{}{} GENERATED FIGURES OF {} {}{}".format(utils.asterisks, utils.asterisks,
                                                             self.parameters['filename_snapshot'].upper(),
                                                             utils.asterisks, utils.asterisks))
            data_analysis.graph_metrics(label=False)

    def __infer_implementation(self, config: dict):
        """
        Decide how we want to do this.

        config is a dict with parameters for tuning the implementation inference
        procedure
        :param config:
        """

    @staticmethod
    def __check_balance_config(config):
        """
        Checks the distribution used to assign the balance

        :param config:
        :return:
        """
        assert "name" in config, "No distribution specified"
        assert config["name"] in ["const", "unif", "normal", "exp", "beta"], "Unrecognized distribution name"

    def __assign_rand_balances(self, config: dict):
        """
        Randomly assigns balances to the channels following the specified distribution.
        Balances are not assigned if config is None.

        :param config: dict, distribution name (key 'name'), and params (keys depend on distribution name).
            Recognized keys are:

            name:   "const", "unif", "normal", "exp", "beta"
            mu:     float (only for name = normal)
            sigma:  float (only for name = normal)
            l:      float (only for name = exp)
            alpha:  float (only for name = beta)
            beta:   float (only for name = beta)

            Examples:
                config = {"name": "const"}
                config = {"name": "unif"}
                config = {"name": "normal", "mu": 0.5, "sigma": 0.2}
                config = {"name": "exp", "l": 1}
                config = {"name": "beta", "alpha": 0.25, "beta": 0.25}
        """

        rand_func = None
        if config is None:
            # Do not assign balances if config is None
            print("INFO: balances not assigned")
            return

        self.__check_balance_config(config)
        print("INFO: balances assigned using a {} distribution ({})".format(config["name"], config))

        if config["name"] == "const":
            rand_func = lambda exp: int(exp[2]["capacity"] / 2)
        elif config["name"] == "unif":
            rand_func = lambda exp: int(np.random.uniform(0, exp[2]["capacity"]))
        elif config["name"] == "normal":
            mu, sigma = config["mu"], config["sigma"]

            def rand_func(exp):
                r = np.random.normal(mu, sigma)
                while r < 0 or r > 1:
                    r = np.random.normal(mu, sigma)
                return exp[2]["capacity"] - int(exp[2]["capacity"] * r)
        elif config["name"] == "exp":
            l_param = config["l"]

            def rand_func(exp):
                r = np.random.exponential(l_param)
                while r > 1:
                    r = np.random.exponential(l_param)
                return exp[2]["capacity"] - int(exp[2]["capacity"] * r)
        elif config["name"] == "beta":
            alpha, beta = config["alpha"], config["beta"]
            rand_func = lambda exp: exp[2]["capacity"] - int(exp[2]["capacity"] * np.random.beta(alpha, beta))

        # TODO: Improve this code. Now we have a mapping between both graphs, so there is no need to
        # store channels already assigned (we can iterate by G1's edges)
        # We randomly assign one of the channel's balances, and set the other balance to the remaining amount
        assigned_channels = {}
        for e in self.g2.edges(data=True):
            if not e[2]["channel_id"] in assigned_channels:
                e[2]["balance"] = rand_func(e)
                assigned_channels[e[2]["channel_id"]] = e[2]["capacity"] - e[2]["balance"]
            else:
                e[2]["balance"] = assigned_channels[e[2]["channel_id"]]

    @staticmethod
    def __check_htlc_config(config):
        """
        Checks the distribution and htlc configuration

        :param config: dict, distribution name (key 'name'), and params (keys depend on distribution name).
        :return:
        """
        assert "name" in config, "No distribution specified"
        assert config["name"] in ["const"], "Unrecognized distribution name"
        if "amount_fract" in config:
            assert config["amount_fract"] * config["number"] <= 1, "Not enough balance for that number of HTLCs!"

    def __assign_rand_htlc(self, config):
        """
        Randomly assigns pending HTLCs to channels following the specified distribution.
        Pending HTLCs are not assigned if config is None.

        :param config: dict, distribution name (key 'name'), and params (keys depend on distribution name).
            Recognized keys are:

            name:           "const"
            number:         int
            amount_fract:   int

            Examples:
                config = {"name": "const", "number": 1, "amount_fract": 0.1}
                """
        rand_func = None
        if config is None:
            # Do not assign balances if config is None
            print("INFO: pending HTLCs not assigned")
            return

        self.__check_htlc_config(config)
        print("INFO: pending HTLCs assigned using a {} distribution ({})".format(config["name"], config))

        if config["name"] == "const":
            def rand_func(exp, rand_seed):
                htlc_dict_temp, amounts_temp = {}, 0
                if self.parameters["is_rand_htlc"]:
                    time_lock_delta = exp[2]['policy_source']['time_lock_delta'] \
                        if 'policy_source' in exp[2] and exp[2]['policy_source'] is not None and \
                           'time_lock_delta' in exp[2]['policy_source'] else 0

                    if self.parameters["is_prand_htlc"]:
                        limit = (1 if config['number'] == 0 else config['number']) if time_lock_delta == 0 \
                            else (int(rand_seed * time_lock_delta))
                        limit = range(1) if limit == 0 else range(limit)
                        timer = int(time_lock_delta/len(limit))
                        timeouts = [int(timer * rand_seed * 2) for i in limit]
                    else:
                        limit = range(1 if config["number"] == 0 else config["number"]) \
                            if time_lock_delta == 0 \
                            else range(int(np.random.uniform(1, time_lock_delta + 1))) \
                            if time_lock_delta < self.parameters['num_pending_htlc'] \
                            else range(int(np.random.uniform(1, self.parameters['num_pending_htlc'] + 1)))

                        timer = int(time_lock_delta / len(limit))
                        timeouts = [int(np.random.uniform(0, timer + 1) * np.random.uniform(1, 2)) for i in limit]
                    balance = exp[2]['balance']
                    for i in limit:
                        amount = int(rand_seed * config["amount_fract"] * balance
                                     if self.parameters["is_prand_htlc"]
                                     else int(np.random.uniform(1, config["amount_fract"] * balance)))
                        if amount < balance:
                            htlc_dict_temp[i] = (amount, timeouts[i])
                            balance -= amount
                            amounts_temp += amount
                        else:
                            amount = int(config["amount_fract"] * balance if self.parameters["is_prand_htlc"]
                                         else np.random.uniform(1, balance))
                            htlc_dict_temp[i] = (amount, timeouts[i])
                            amounts_temp += amount
                            break
                else:
                    for i in range(config["number"]):
                        amount = config["amount_fract"] * exp[2]["balance"]
                        htlc_dict_temp[i] = (amount, 0)
                        amounts_temp += amount
                return htlc_dict_temp, amounts_temp

        np.random.seed(29)
        numbs = [0.25, 0.5]
        numbs.extend([x for x in (abs(np.random.randn(100))) if x * 144 < 14][:3])
        seeds = cycle(numbs)
        for e in self.g2.edges(data=True):
            htlc_dict, amounts = rand_func(e, next(seeds))
            e[2]["pending_htlc"] = htlc_dict
            e[2]["balance"] = e[2]["balance"] - amounts

    def __check_correctness(self, is_mesage=True):
        """
        Check the three restrictions explained in the paper (page 2)

        :return:
        Prints info about Balance and Capacity on the channels
        """
        print("INFO: checking correctness of the imported graph (disable for better performance)")

        # Check 1: Same number of nodes in both graphs
        assert self.g1.number_of_nodes() == self.g2.number_of_nodes()

        # Check 2: Double number of edges in g2
        assert 2 * self.g1.number_of_edges() == self.g2.number_of_edges()

        # Check 3: The sum of the balances and blocked amounts in HTLCs on both sides of the channel
        # must be equal to the capacity
        for e in self.g1.edges(data=True, keys=True):
            r = self.get_ke2_from_ke1(e[2], u=e[0], v=e[1])

            one_edge_data = self.g2[e[0]][e[1]][r[0]]
            other_edge_data = self.g2[e[1]][e[0]][r[1]]

            if is_mesage:
                print('CHANNEL_ID: %s' % (e[2]))
            balance_one_edge_data = one_edge_data["balance"] + sum([v[0]
                                                                    for v in one_edge_data["pending_htlc"].values()])
            if is_mesage:
                print('%sBALANCE: %s AND BALANCE SQUARED: %s FROM %s TO %s' % (utils.spaces, one_edge_data["balance"],
                                                                           balance_one_edge_data,
                                                                           self.nodeDict[e[0]]['alias'],
                                                                           self.nodeDict[e[1]]['alias']))

            balance_other_edge_data = other_edge_data["balance"] + sum([v[0]
                                                                        for v in
                                                                        other_edge_data["pending_htlc"].values()])
            if is_mesage:
                print('%sBALANCE: %s AND BALANCE SQUARED: %s FROM %s TO %s' % (utils.spaces, other_edge_data["balance"],
                                                                           balance_other_edge_data,
                                                                           self.nodeDict[e[1]]['alias'],
                                                                           self.nodeDict[e[0]]['alias']))
            if is_mesage:
                print('%sCAPACITY CHANNEL: %s' % (utils.spaces, e[3]["capacity"]))

            assert balance_one_edge_data + balance_other_edge_data == e[3]["capacity"]

    def get_ke2_from_ke1(self, ke1, u=None, v=None):
        """
        Given the key of an undirected edge from G1, return the two keys corresponding to the directed edges in G2.

        :param ke1:key of an edge from G1
        :param u: a node incident to the edge
        :param v: the other node incident to the edge
        :return:
        """
        # TODO: Is there a better way to do this? We need to know the nodes in order to retrieve the edge!
        if u is None or v is None:
            for e in self.g1.edges(keys=True, data=True):
                if e[2] == ke1:
                    u, v = e[0], e[1]
                    break

        ke2_1 = "{}-{}".format(ke1, u)
        ke2_2 = "{}-{}".format(ke1, v)
        return ke2_1, ke2_2

    @staticmethod
    def get_ke1_from_ke2(ke2):
        """
        Given the key of a directed edge from G2, return the key from the corresponding undirected edge from G1.

        :param ke2: key of an edge from G2
        :return: key of an edge from G1
        """
        return ke2.split("-")[0]

    def get_number_of_nodes(self):
        """
        :return: int
        """
        return self.g1.number_of_nodes()

    def get_total_number_of_channels(self):
        """
        :return: int
        """

    def get_number_of_channels_by_node(self, node=None):
        """
        :return: dictionary with node ids as keys, number of channels as values if node=None,
            or int with number of channels by a given node
        """
        if node is None:
            dict_node_edge = {}
            for element in self.g1.nodes:
                dict_node_edge[element] = len(self.g1.edges(element))
            return dict_node_edge
        else:
            return len(self.g1.edges(node))

    def get_number_of_channels_distr(self, normalized=True):
        """
        Return the distribution of the number of channels per the node (both the pdf and cdf). The format can be:
            x: a list with all number of channels found
            pdf: a list with the number of nodes with each of the number of channels in x
            cdf: a list with the number of nodes with less than or equal each of the number of channels in x
        If normalized=True, return pdf and cdf values over 1. Otherwise, use absolute numbers.

        :param normalized: boolean
        :return: 3-element tuple, each element is a list
        """
        pass

    def get_total_network_capacity(self):
        """
        :return: int
        """
        total_capacity = 0
        for i in self.g1.edges:
            total_capacity = total_capacity + self.g1[i[0]][i[1]][i[2]]['capacity']
        return total_capacity

    def get_network_capacity_by_node(self, node=None):
        """
        :return: dictionary with node ids as keys, capacity per node if node=None,
            or int with capacity by a given node
        """
        if node is None:
            dict_cap_node = {}
            for e in self.g1.nodes:
                dict_cap_node[e] = 0
            for i in self.g1.edges:
                dict_cap_node[i[0]] = dict_cap_node[i[0]] + self.g1[i[0]][i[1]][i[2]]['capacity']
                dict_cap_node[i[1]] = dict_cap_node[i[1]] + self.g1[i[0]][i[1]][i[2]]['capacity']
            return dict_cap_node
        else:
            capacity = 0
            for i in self.g1.edges:
                if (i[0] == node) or (
                        i[1] == node):
                    capacity = capacity + self.g1[i[0]][i[1]][i[2]]['capacity']
            return capacity

    def get_network_capacity_distr(self, normalized=True):
        """
        COMPTE: això ho faria sobre les arestes! (pensem si també té sentit tenir les dades sobre els nodes)

        Return the distribution of the capacity (both the pdf and cdf). The format can be:
            x: a list with all capacities found
            pdf: a list with the number of nodes with each of the capacities in x
            cdf: a list with the number of nodes with less than or equal each of the capacities in x

        If normalized=True, return pdf and cdf values over 1. Otherwise, use absolute numbers.

        :return: 3-element tuple, each element is a list
        """

    def get_total_disabled_capacity(self):
        """
        Dependent on node policy and balance.

        :return: tuple, (total disabled, percentage over the total)
        """
        counter = 0
        for i in self.g2.edges:
            if (self.g2[i[0]][i[1]][i[2]]['policy_dest'] is not None) and (
                    self.g2[i[0]][i[1]][i[2]]['policy_dest']['disabled']):
                counter = counter + self.g2[i[0]][i[1]][i[2]]['balance']
        return counter

    def get_disabled_capacity_by_node(self, node=None):
        """
        :return: dictionary with node ids as keys, disabled capacity per node if node=None,
            or int with disabled capacity by a given node
        """
        if node is None:
            dict_cap_node = {}
            for e in self.g2.nodes:
                dict_cap_node[e] = 0
            for i in self.g2.edges:
                if (self.g2[i[0]][i[1]][i[2]]['policy_dest'] is not None) and (
                        self.g2[i[0]][i[1]][i[2]]['policy_dest']['disabled']):
                    dict_cap_node[i[0]] = dict_cap_node[i[0]] + self.g2[i[0]][i[1]][i[2]]['balance']
            return dict_cap_node
        else:
            counter = 0
            for i in self.g2.edges:
                if (i[0] == node) and (self.g2[i[0]][i[1]][i[2]]['policy_dest'] is not None) and (
                        self.g2[i[0]][i[1]][i[2]]['policy_dest']['disabled']):
                    counter = counter + self.g2[i[0]][i[1]][i[2]]['balance']
            return counter

    def get_disabled_capacity_distr(self, normalized=True):
        pass

    def get_total_blocked_amount(self):
        """
        Return the total blocked amount in HTLCs

        :return: tuple, (total blocked, percentage over the total)
        """
        pass

    def get_blocked_amount_by_node(self):
        pass

    def get_total_blocked_distr(self, normalized=True):
        """
        Return the distribution of the blocked capacity (both the pdf and cdf). The format can be:
            x: a list with all blocked capacities found
            pdf: a list with the number of nodes with each of the blocked capacities in x
            cdf: a list with the number of nodes with less than or equal each of the blocked capacities in x
        If normalized=True, return pdf and cdf values over 1. Otherwise, use absolute numbers.

        :return: 3-element tuple, each element is a list
        """

    def get_total_useful_capacity(self):
        """
        total - disabled - blocked in htlc

        :return: tuple, (total useful, percentage over the total)
        """
        pass

    def get_useful_capacity_by_node(self):
        pass

    def get_useful_capacity_distr(self, normalized=True):
        pass

    def get_balance_by_node(self, node=None):
        """
        Delivers the balance of either the whole network or a specific node

        :param node: node to get balance
        :return: dictionary with node ids as keys, balance per node if node=None,
            or int with balance by a given node
        """
        if node is None:
            dict_balance_node = {}
            for e in self.g2.nodes:
                dict_balance_node[e] = 0
            for i in self.g2.edges:
                dict_balance_node[i[0]] = dict_balance_node[i[0]] + self.g2[i[0]][i[1]][i[2]]['balance']
            return dict_balance_node
        else:
            balance = 0
            for i in self.g2.edges:
                if i[0] == node:
                    balance = balance + self.g2[i[0]][i[1]][i[2]]['balance']
            return balance

    def get_balance_distr(self, normalized=True):
        """
        Return the distribution of the balances (both the pdf and cdf). The format can be:
            x: a list with all balances found
            pdf: a list with the number of nodes with each of the balances in x
            cdf: a list with the number of nodes with less than or equal each of the balances in x
        If normalized=True, return pdf and cdf values over 1. Otherwise, use absolute numbers.

        :return: 3-element tuple, each element is a list
        """

    def get_number_of_nodes_by_implementation(self, divide_by_version=False):
        """
        Count number of nodes for each implementation. Depending on the flag divide_by_version,
        we distinguish between different versions of the same implementation or not.

        :return: dict, key is implementation name, value is tuple with number of nodes and percentage over the total
        """
        pass

    def get_implementation_by_node(self, node=None):
        pass


logging.basicConfig(filename='{:%Y-%m-%d-%H:%M}.log'.format(datetime.now()),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)

logger = logging.getLogger('mylog')
logging.info("Starting")

json_filename = 'lngraph_2018_10_12__12_00.json'
"""
Random balance distributions used in the paper:
    config = {"name": "const"}
    config = {"name": "unif"}
    config = {"name": "normal", "mu": 0.5, "sigma": 0.2}
    config = {"name": "exp", "l": 1}
    config = {"name": "beta", "alpha": 0.25, "beta": 0.25}
"""
balance_config = {"name": "const"}
"""
Example of random pending HTLCs distributions:
    config = {"name": "const", "number": 1, "amount_fract": 0.1}
    config = {"name": "const", "number": 0}
We only have a constant distribution, that assigns the given number of pending HTLC to every channel, with
an amount_fract of the amount of the balance locked in each HTLC.
"""
htlc_config = {"name": "const", "number": 0, "amount_fract": 0.1}

ln_graph = LNCentrality(json_filename, balance=balance_config, htlc=htlc_config)

# print(nx.info(ln_graph.g1))
# print(nx.info(ln_graph.g2))

logging.shutdown()
