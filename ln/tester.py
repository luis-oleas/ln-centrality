import pickle
import pylab as p
import networkx as nx
from networkx import Graph

import ln.utils as utils
import matplotlib.pyplot as plt
from ln.connector import lnd_client as lnd, clightning_client as clight, eclair_client as eclair

import datetime
import random


class Tester:
    def __init__(self):
        # self.check_numb_nodes()
        # self.send_payment_lnd()
        # self.describe()
        # self.save_pickle()
        self.load_pickle()
        # self.describe()

    def check_numb_nodes(self):
        data = utils.load_file("/home/pis1901/PycharmProjects/ln_model/ln/data/snapshot",
                               "lngraph_2019_02_25__04_00.json", False)
        count = 0
        for k, v in data.items():
            if k != 'message' and k != "edges":
                count = count + len(v)

        print(count)

    def save_pickle(self):
        G = nx.MultiGraph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=2)
        G.add_edge(2, 1, weight=1)
        G.add_edge(2, 3, weight=3)

        pickled_file = open('pickled_file.pickle', 'wb')
        pickle.dump(G, pickled_file)
        pickled_file.close()

    def load_pickle(self):
        infile = open('/home/pis1901/PycharmProjects/ln_model/ln/data/results/lngraph_2019_02_25__11_00_G1.pickle', 'rb')
        data = pickle.load(infile)
        infile.close()

        nx.draw(data)
        p.show()

        return data

    def describe(self):
        global describeNetwork
        '''
        Describe the network: degrees, clustering, and centrality measures
        '''
        # Degree
        # The number of connections a node has to other nodes.
        G = nx.MultiDiGraph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=2)
        G.add_edge(2, 1, weight=1)
        G.add_edge(2, 3, weight=3)
        G.add_edge(2, 0, weight=3)
        G.add_edge(0, 3, weight=3)
        G.add_edge(3, 5, weight=3)
        degrees = nx.degree(G)
        all_paths = []
        all_edges = []
        for v, d in G.in_degree():
            if d == 0:
                print(d)
        roots = list((v for v, d in G.in_degree()))
        leaves = list((v for v, d in G.out_degree()))

        # roots = G.nodes()
        # leaves = G.nodes()
        for root in roots:
            print('root:' + str(root))
            for leaf in leaves:
                print('     leaf:' + str(leaf))
                paths = nx.all_simple_paths(G, source=root, target=leaf)
                for path in paths:
                    if len(path) > 2:
                        all_paths.append(path)
                        for i in range(len(path)):
                            if i != 0 and i % 2 == 0:
                                edge = [path[i-1], path[i]]
                                if edge not in all_edges:
                                    all_edges.append(edge)
        print(all_paths)
        print(all_edges)

        G = nx.Graph([(0, 1), (1, 2), (0, 3), (3, 2), (3, 0)])
        roots = G.nodes()
        leaves = G.nodes()
        all_paths = []
        all_edges = []
        for root in roots:
            for leaf in leaves:
                paths = nx.all_simple_paths(G, source=root, target=leaf)
                for path in paths:
                    if len(path) > 2:
                        all_paths.append(path)
        print(all_paths)
        # [[0, 1, 2], [0, 3, 2]]
        print(nx.dag_longest_path(G))

        betweeness = nx.betweenness_centrality(G, weight='weight', normalized=True, k=3)
        for k, v in betweeness.items():
            print("%s: %s" % (k, v))

        G = G.to_undirected()
        # Node centrality measures
        FCG = list(G.subgraph(c) for c in nx.connected_components(G))
        for comp in FCG:
            betweeness = nx.current_flow_betweenness_centrality(comp)
            for k, v in betweeness.items():
                print("%s: %s" % (k, v))

    def graph_metric_degree_strength(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=2)
        G.add_edge(2, 1, weight=1)
        G.add_edge(2, 3, weight=3)

        degree_weight = G.out_degree(weight='weight')
        for node, degree in G.out_degree():
            print(node, degree, degree_weight[node], 0 if degree == 0 else round(degree * pow(degree_weight[node]/degree, 1.5), 3))

    def get_info_lnd(self):
        info = lnd.get_info(host="localhost", port=10001,
                            macaroon="/home/deic/.polar/networks/1/volumes/lnd/alice/data/chain/bitcoin/regtest/admin.macaroon",
                            cert="/home/deic/.polar/networks/1/volumes/lnd/alice/tls.cert")

        return info

    def get_query_routes_clight(self):
        pay = clight.query_routes(
            macaroon_dir="/home/deic/.polar/networks/1/volumes/c-lightning/ivan/lightningd/regtest/lightning-rpc",
            node_origin='ivan', node_destiny='alice', amount=1000)
        return pay

    def get_query_routes_eclair(self):
        pay = eclair.query_routes(node_destiny='niaj', amount=1000, host="localhost", port=8290, user='',
                                  passwd="eclairpw")
        return pay

    def get_info_eclair(self):
        info = eclair.get_info("localhost", 8283, "", "eclairpw")

        return info

    def send_payment_lnd(self):
        pub_key_destiny = "026948dcad0d691137c3b052c43bc791e085853846b0b40f4dd47e38a0264b1bcb"
        lnd.send_payment_rpc(
            macaroon_dir="/home/deic/.polar/networks/1/volumes/lnd/alice/data/chain/bitcoin/regtest/admin.macaroon",
            cert_dir="/home/deic/.polar/networks/1/volumes/lnd/alice/tls.cert",
            payment_hash=utils.request_payment_hash_destiny(pub_key_destiny=pub_key_destiny)[0],
            pubkey_destiny=pub_key_destiny, final_cltv_delta=120, payment_amount=1500, port=10001, host='127.0.0.1')

        lnd.send_payment_router(
            macaroon_dir="/home/deic/.polar/networks/1/volumes/lnd/alice/data/chain/bitcoin/regtest/admin.macaroon",
            cert_dir="/home/deic/.polar/networks/1/volumes/lnd/alice/tls.cert",
            payment_hash=utils.request_payment_hash_destiny(pub_key_destiny=pub_key_destiny)[0],
            pubkey_destiny=pub_key_destiny, final_cltv_delta=120, payment_amount=1500, port=10001, host='127.0.0.1')


time = datetime.datetime.now()
# print(time)
str_time = datetime.datetime.now().strftime("%Y/%m/%dT%H:%M:%S")
print(str_time)

print("******************************************************************************************************")
print(Tester())
