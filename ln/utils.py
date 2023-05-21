import re
import os
import sys
import math
import glob
import json
import pickle
import ntpath
import logging
import hashlib
import ipaddress
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Any, List
from ln.centrality.closeness import closeness_centrality
from ln.centrality.betweenness import betweenness_centrality
from ln.centrality.flow_betweenness import flow_betweenness_centrality, flow_debug
from ln.centrality.current_flow_betweenness import current_flow_betweenness_centrality

spaces = "".rjust(5)
asterisks = "**********"


def input_value(default: str, message: str, is_path: bool, is_value: bool):
    """
    Validates input data used to perform query routes. The data to control is of type integer and directory to load
    the json files


    :param default: default value to return in case of a None or empty value
    :param message: message to be printed as informative
    :param is_path: indicates if the value is a path/directory from which the simulation will load the json files
    :param is_value: indicates if the value is a number such as the amount in satoshis
    :return: the validated input data
    """
    data = None
    if is_value:
        while True:
            try:
                data = input(message)
                if data != '':
                    data = int(data)
            except ValueError:
                print("Expected an int value!!!!")
                continue
            else:
                break
    else:
        data = input(message)

    if is_path:
        has_error = False
        while True:
            if has_error:
                data = input(message)
            try:
                with open(data):  # OSError if file does not exist or is invalid
                    break
            except OSError as err:
                if len(data) > 0:
                    print("OS error: {0}".format(err))
                    has_error = True
                    continue
                else:
                    break
            except ValueError:
                if data == '':
                    print("Unexpected error:", sys.exc_info()[0])
                    has_error = True
                    continue
                else:
                    break

    return default if is_value and data == 0 or data == '' else data


def check_preimage_hash(preimage, hash_value) -> bool:
    """
    Validates that the preimage of the hash corresponds to the one provided on the hop to perform the payment

    :param preimage:generated random number
    :param hash_value:hash value of the preimage provided by the hop
    :return: true or false based on the operation
    """
    if hashlib.sha256(str(preimage).encode()).hexdigest() == hash_value:
        return True
    return False


def request_payment_hash_destiny(pub_key_destiny: str):
    """
    Generates a preimage and its hash value based on the pub_key of the destiny

    :param pub_key_destiny: pub_key of the node destiny
    :return: preimage and hash value
    """
    num = int(re.sub('[^0-9_]', '', str(pub_key_destiny)))
    preimage = np.random.uniform(0, num)
    return hashlib.sha256(str(preimage).encode()).hexdigest(), preimage


def validate_ip(host: str, ip: str) -> str:
    """
    Validates the ip/host of the node to connect

    :param host: host of the node
    :param ip: ip of the node
    :return: ip validated
    """
    try:
        host = host if ip == '' else ip
        ip = ipaddress.ip_address(host)
        print('%s is a correct IP%s address.' % (ip, ip.version))
    except ValueError:
        print('address/netmask is invalid: %s. Default ip  used: %s' % (ip, host))
        ip = host

    return ip


def load_pickle(location: str, filename: str):
    """
    Lets to load a pickle file

    :param location: directory in which the file is located
    :param filename: name of the file
    """
    path = os.path.join(location, filename + '.pickle')
    infile = open(path, 'rb')
    data = pickle.load(infile)
    infile.close()

    return data


def load_file(location: str, file_name: str, is_snapshot: bool):
    """
    Lets to load a file by its name from a location in the project

    :param location: directory in which the file is located
    :param file_name: name of the file
    :param is_snapshot: indicates that the file is a snapshot and set the nodes and edges
    :return: data stored on the file
    """
    with open(os.path.join(location, file_name), encoding="utf8") as f:
        if is_snapshot:
            data = []
            values = {'filename_snapshot': file_name, 'data': set_data_nodes_edges(json.load(f), False)}
            data.append(values)
        else:
            data = json.load(f)

    return data


def select_files(starting_file, periodicity, this_file):
    PERIODICITY = ["all", "a", "yearly", "y", "monthly", "m", "weekly", "w", "daily", "d"]
    DATE_FORMAT = "lngraph_%Y_%m_%d__%H_00.json.zst"

    if periodicity not in PERIODICITY:
        raise Exception("Periodicity not recognized")
    try:
        dt_start = datetime.strptime(starting_file, DATE_FORMAT)
        dt_current = datetime.strptime(this_file, DATE_FORMAT)
    except ValueError:
        return False

    while dt_start < dt_current:
        if periodicity in ["all", "a"]:
            return True
        if periodicity in ["yearly", "y"]:
            dt_start += timedelta(days=365)
        if periodicity in ["monthly", "m"]:
            dt_start += timedelta(days=30)
        if periodicity in ["weekly", "w"]:
            dt_start += timedelta(days=7)
        if periodicity in ["daily", "d"]:
            dt_start += timedelta(days=1)
    return dt_start == dt_current


def load_set_files(location: str, extension: str):
    """

    :param location: directory in which the files are located
    :param extension: extension of the file to open
    :return: data stored on the file
    """
    data = [] if extension == '*.json' else {}
    for path_filename in glob.glob(os.path.join(location, extension)):
        if True:
            logging.debug("Loading {}".format(path_filename))
            with open(path_filename, 'r', encoding="utf8") as file_data:
                raw_data = json.load(file_data)
                file_name = ntpath.basename(path_filename)
                if extension == '*.json':
                    values = {'filename_snapshot': file_name, 'data': set_data_nodes_edges(raw_data, False)}
                    data.append(values)
                else:
                    if 'metrics_' in file_name:
                        data = set_data_metrics(file_name, raw_data)

    return data


def save_pickle(location: str, filename: str, data: Any):
    """
    Lets to load a pickle file

    :param location: directory in which the file is located
    :param filename: name of the file
    :param data: data to save in the file
    """
    path = os.path.join(location, filename)
    pickled_file = open(path, 'wb')
    pickle.dump(data, pickled_file)
    pickled_file.close()


def save_file(location: str, file_name: str, data, has_datetime: bool = True):
    """
    Let store data on a file

    :param location: directory in which the file is located
    :param file_name: name of the file
    :param data: data to be stored in format dictionary
    :param has_datetime: flag that indicates that the name of the file contains the datetime in epoch style
    :return:
    """
    if has_datetime:
        time_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        temp = file_name.split('.')
        file_name = temp[0] + '_' + time_str + '.' + temp[1]

    with open(os.path.join(location, file_name), 'w') as fp:
        data_json = json.loads(data)
        json.dump(data_json, fp, indent=4)
        fp.close()

    # temp_file = open(os.path.join(location, file_name), 'w')
    # with temp_file as fp:
    #     json.dump(data, fp)
    # temp_file.close()
    #
    # temp_file = open(os.path.join(location, file_name), 'r')
    # for line in temp_file:
    #     # read replace the string and store on a variable
    #     data_json = line.replace('\\', '').replace('"{', '{').replace('}"', '}')
    # data_json = json.loads(data_json)
    #
    # time_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    # temp = file_name.split('.')
    # file_name = temp[0] + '_' + time_str + '.' + temp[1]
    # final_file = open(os.path.join(location, file_name), 'w')
    # with final_file as fp:
    #     json.dump(data_json, fp, indent=4)
    # temp_file.close()
    # final_file.close()


def set_data_nodes_edges(data: nx, is_message: bool = True) -> nx:
    """
    Sets the structure of the snapshot divided on nodes and edges. Moreover, it set the value of those parameters that
    are either None or empty from the edges and nodes

    :param data: data gathered from the json file
    :param is_message: meessage to print
    :return: data validated and set
    """
    index = Counter()
    dict_pub_key = {}
    for node in data['nodes']:
        index.preinc()
        if 'last_update' not in node: node['last_update'] = 0
        if 'alias' not in node: node['alias'] = ''
        if 'addresses' not in node: node['addresses'] = []
        if 'color' not in node: node['color'] = '#000000'
        if 'features' not in node: node['features'] = {}
        dict_pub_key[node['pub_key']] = node['alias']
        if is_message:
            print('{}INFO: Node #{} - alias: {} - pub_key: {}'.format(spaces, index.v, node['alias'],
                                                                      node['pub_key']))
    # if is_message:
    #    input("Press ENTER to continue.....")
    index.set(0)
    for edge in data['edges']:
        index.preinc()
        if 'last_update' not in edge: edge['last_update'] = 0
        if 'node1_policy' in edge and edge['node1_policy'] is not None:
            policy1 = edge['node1_policy']
            if 'disabled' not in policy1: policy1['disabled'] = True
        if 'node2_policy' in edge and edge['node2_policy'] is not None:
            policy2 = edge['node2_policy']
            if 'disabled' not in policy2: policy2['disabled'] = True

        # if 'node1_policy' not in edge or edge['node1_policy'] is None:
        #     edge['node1_policy'] = {'min_htlc': '1000', 'fee_base_msat': '1', 'time_lock_delta': 144,
        #                          'fee_rate_milli_msat': 1000}
        # else:
        #     policy = edge['node1_policy']
        #     if 'min_htlc' not in policy: policy['min_htlc'] = '1000'
        #     if 'disabled' not in policy: policy['disabled'] = False
        # if 'node2_policy' not in edge or edge['node2_policy'] is None:
        #     edge['node2_policy'] = {'min_htlc': '1000', 'fee_base_msat': '1', 'time_lock_delta': 144,
        #                          'fee_rate_milli_msat': 1000}
        # else:
        #     policy = edge['node2_policy']
        #     if 'min_htlc' not in policy: policy['min_htlc'] = '1000'
        #     if 'disabled' not in policy: policy['disabled'] = False

        # if 'node1_policy' in edge:
        #     policy = edge['node1_policy']
        #     if policy is not None and 'min_htlc' not in policy: policy['min_htlc'] = '1000'
        # if 'node2_policy' in edge:
        #     policy = edge['node2_policy']
        #     if policy is not None and 'min_htlc' not in policy: policy['min_htlc'] = '1000'

        if is_message:
            print('{}INFO: Channel #{}({}) - from {} ({}) to {} ({})'.format(spaces, index.v,
                                                                             edge['channel_id'],
                                                                             dict_pub_key[edge['node1_pub']],
                                                                             edge['node1_pub'],
                                                                             dict_pub_key[edge['node2_pub']],
                                                                             edge['node2_pub']))

    return data


def set_data_metrics(file_name: str, raw_data: dict):
    data = {}
    date = file_name.replace('metrics_', '').replace('.json', '')
    for cc_k, cc_v in raw_data.items():
        if cc_k != 'message':
            if cc_k not in data:
                data[cc_k] = {}
                for node_k, node_v in cc_v.items():
                    data[cc_k].update({node_k: {date: node_v}})
            else:
                for node_k, node_v in cc_v.items():
                    if node_k not in data[cc_k]:
                        data[cc_k][node_k] = {date: node_v}
                    else:
                        data[cc_k][node_k].update({date: node_v})

    return data


def get_nodes_longest_paths(g: nx, parameters: dict) -> List[List]:
    all_paths = []
    all_edges = []
    roots = list((v for v, d in g.in_degree()))
    leaves = list((v for v, d in g.out_degree()))

    for root in roots:
        for leaf in leaves:
            paths = nx.all_simple_paths(g, source=root, target=leaf)
            for path in paths:
                if len(path) > parameters['length_path']:
                    all_paths.append(path)
                    for i in range(len(path)):
                        if i != 0 and i % parameters['length_path'] == 0:
                            edge = [path[i - 1], path[i]]
                            if list(edge) not in all_edges:
                                all_edges.append(list(edge))

    return all_edges


def set_similar_links(e, links_nodes, attr, is_reverse, links=None, links_pairs=None, prop='fee'):
    """

    Parameters
    ----------
    e: contains the data of the parameter
    links_nodes: links between a pair of nodes to validate according to the edge property (fee/capacity)
    attr: the final attribute to be set to the edge
    is_reverse: parameter used to specify the direction of the edge e1 -> e2 or e2 -> e1
    links: groups the channels by its id that belongs to a pair of nodes
    links_pairs: groups the channels by pairs of nodes
    prop: specify whether the property of the edge is fee or capacity

    Returns
    -------

    """
    key = '{}-{}'.format(e['node1_pub'], e['node2_pub']) \
        if not is_reverse else '{}-{}'.format(e['node2_pub'], e['node1_pub'])
    if key not in links_nodes:
        links_nodes[key] = attr
    else:
        if prop == 'fee':
            if links_nodes[key]['fee'] is None:
                links_nodes[key] = attr
            elif attr['fee'] is not None:
                links_nodes[key] = attr if attr['fee'] < links_nodes[key]['fee'] else links_nodes[key]
        elif prop == 'capacity':
            links_nodes[key] = attr if attr['capacity'] > links_nodes[key]['capacity'] else links_nodes[key]

    disabled_channel = False if e['node1_policy'] is not None and 'disabled' in e['node1_policy'] and \
                                not e['node1_policy']['disabled'] and e['node2_policy'] is not None and \
                                'disabled' in e['node2_policy'] and not e['node2_policy']['disabled'] else True
    if links is not None:
        if key not in links:
            # links[key] = [e['channel_id']]
            links[key] = {"channels": [e['channel_id']], "disabled": 1 if disabled_channel else 0}
        else:
            # links.setdefault(key, []).append(e['channel_id'])
            links[key]["channels"].append(e['channel_id'])
            links[key]["disabled"] = links[key]["disabled"] + 1 if disabled_channel else links[key]["disabled"]

    key_node1 = e['node1_pub']
    key_node2 = e['node2_pub']
    if links_pairs is not None:
        channel = {"channel": e['channel_id'], "capacity": e['capacity'],
                   "fee": None if e['node1_policy'] is None else e['node1_policy']['fee_base_msat']}
        if key_node1 not in links_pairs:
            links_pairs[key_node1] = {key_node2: {"disabled": [channel] if disabled_channel else [],
                                                  "enabled": [] if disabled_channel else [channel]}}
        elif key_node2 not in links_pairs[key_node1]:
            links_pairs[key_node1][key_node2] = {"disabled": [channel] if disabled_channel else [],
                                                 "enabled": [] if disabled_channel else [channel]}
        else:
            if disabled_channel:
                links_pairs[key_node1][key_node2]["disabled"].append(channel)
            else:
                links_pairs[key_node1][key_node2]["enabled"].append(channel)

    attr = links_nodes[key]
    return attr, links_nodes, links, links_pairs


def validate_weight_policy(weight_policy_source, weight_policy_dest):
    if weight_policy_source is None:
        if weight_policy_dest is None:
            weight_policy = None
        else:
            weight_policy = weight_policy_dest
    else:
        if weight_policy_dest is None:
            weight_policy = weight_policy_source
        elif weight_policy_source < weight_policy_dest:
            weight_policy = weight_policy_source
        else:
            weight_policy = weight_policy_dest

    return weight_policy


def populate_graphs(data: nx):
    """
    NODES: Read the JSON file and import all node data to the g1_unrestricted and g2_unrestricted graph.
    EDGES: Read the JSON file and import all edge data to the g1_unrestricted and g2_unrestricted graph.

    :param data: data load from json file
    :return: g1_unrestricted, g2_unrestricted, nodeDict, edgeDict
    """
    g1_unrestricted = nx.MultiGraph()
    g1_restricted = nx.MultiGraph()
    gs1_unrestricted = nx.DiGraph()
    gs1_restricted = nx.DiGraph()
    g2_unrestricted = nx.MultiDiGraph()
    g2_restricted = nx.MultiDiGraph()
    g2_restricted_path = nx.MultiDiGraph()
    gs2_unrestricted = nx.DiGraph()
    gs2_restricted = nx.DiGraph()
    # NODES: Read the JSON file and import all node data to the g1_unrestricted graph.
    for n in data['nodes']:
        g1_unrestricted.add_node(n['pub_key'], last_update=n['last_update'],
                                 alias=n['alias'], addresses=n['addresses'], color=n['color'],
                                 features=n['features'])
        g1_restricted.add_node(n['pub_key'])
        gs1_unrestricted.add_node(n['pub_key'])
        gs1_restricted.add_node(n['pub_key'])

    node_dict = dict(g1_unrestricted.nodes(data=True))
    total_capacity_restricted_g1 = 0
    total_capacity_unrestricted_g1 = 0
    count_disabled_g1 = Counter()
    count_enabled_g1 = Counter()
    # EDGES: Read the JSON file and import all edge data to the g1_unrestricted graph.
    for e in data['edges']:
        total_capacity_unrestricted_g1 += int(e['capacity'])

        weight_policy_source = int(e['node1_policy']['fee_base_msat']) \
            if 'node1_policy' in e and e['node1_policy'] is not None else None
        weight_policy_dest = int(e['node2_policy']['fee_base_msat']) \
            if 'node2_policy' in e and e['node2_policy'] is not None else None

        weight_policy = validate_weight_policy(weight_policy_source, weight_policy_dest)

        attr = {'chan_point': e['chan_point'], 'last_update': e['last_update'],
                'node1_pub': e['node1_pub'], 'node2_pub': e['node2_pub'],
                'capacity': int(e['capacity']), 'fee_source': weight_policy_source, 'fee_dest': weight_policy_dest,
                'fee': weight_policy, 'policy_source': e["node1_policy"], 'policy_dest': e["node2_policy"]}
        g1_unrestricted.add_edge(e['node1_pub'], e['node2_pub'], key=e['channel_id'], **attr)

        if e['node1_policy'] is not None and 'disabled' in e['node1_policy'] and not e['node1_policy']['disabled'] and \
                e['node2_policy'] is not None and 'disabled' in e['node2_policy'] and not e['node2_policy']['disabled']:
            count_enabled_g1.preinc()
            total_capacity_restricted_g1 += int(e['capacity'])
            g1_restricted.add_edge(e['node1_pub'], e['node2_pub'], key=e['channel_id'], **attr)
        else:
            count_disabled_g1.preinc()

    print("TOTAL CAPACITY G1: \n\tUnrestricted:{}\n\tRestricted:{}".format(total_capacity_unrestricted_g1,
                                                                           total_capacity_restricted_g1))
    print("INFO:\n\tChannels disabled:{}\n\tChannels enabled:{}\n\t\tTotal number of edges unrestricted:{}"
          "\n\t\tTotal number of nodes unrestricted:{}".
          format(count_disabled_g1.v, count_enabled_g1.v, len(g1_unrestricted.edges),
                 len(g1_unrestricted.nodes)))

    edge_dict = {}
    for e in g1_unrestricted.edges(data=True, keys=True):
        edge_dict[e[2]] = e

    # NODES: Read the JSON file and import all node data to the g2_unrestricted graph.
    for n in data['nodes']:
        g2_unrestricted.add_node(n['pub_key'])
        g2_restricted.add_node(n['pub_key'])
        gs2_unrestricted.add_node(n['pub_key'])
        gs2_restricted.add_node(n['pub_key'])
        g2_restricted_path.add_node(n['pub_key'])

    # EDGES: Read the JSON file and import all edge data to the g2_unrestricted graph.
    links_node1_fee = {}
    links_node2_fee = {}
    links_node1_cap = {}
    links_node2_cap = {}
    links = {}
    links_pairs = {}
    extra_nodes = []
    for e in data['edges']:
        weight_policy_source = int(e['node1_policy']['fee_base_msat']) \
            if 'node1_policy' in e and e['node1_policy'] is not None else None
        weight_policy_dest = int(e['node2_policy']['fee_base_msat']) \
            if 'node2_policy' in e and e['node2_policy'] is not None else None
        weight_policy = validate_weight_policy(weight_policy_source, weight_policy_dest)
        k = "{}-{}".format(e['channel_id'], e['node1_pub'])
        attr = {'channel_id': e['channel_id'], 'last_update': e['last_update'],
                'policy_source': e["node1_policy"], 'policy_dest': e["node2_policy"],
                'capacity': int(e['capacity']), 'fee_source': weight_policy_source,
                'fee_dest': weight_policy_dest, 'fee': weight_policy}
        g2_unrestricted.add_edge(e['node1_pub'], e['node2_pub'], key=k,
                                 **attr)  # TODO:(now we can use mappings between graphs, but this may be faster)
        attrib, links_node1_fee, links, links_pairs = set_similar_links(e, links_node1_fee, attr, False, links,
                                                                        links_pairs, prop='fee')
        if attrib['fee'] is not None:
            gs2_unrestricted.add_edge(e['node1_pub'], e['node2_pub'],
                                      key="{}-{}".format(e['node1_pub'], e['node2_pub']), **attrib)

        attrib, links_node1_cap, _, _ = set_similar_links(e, links_node1_cap, attr, False, None, None, prop='capacity')
        if attrib['capacity'] is not None:
            attrib['flow'] = 0
            gs1_unrestricted.add_edge(e['node1_pub'], e['node2_pub'], key="{}-{}".format(e['node1_pub'], e['node2_pub']),
                                      **attrib)

        weight_policy_source = int(e['node2_policy']['fee_base_msat']) \
            if 'node2_policy' in e and e['node2_policy'] is not None else None
        weight_policy_dest = int(e['node1_policy']['fee_base_msat']) \
            if 'node1_policy' in e and e['node1_policy'] is not None else None
        weight_policy = validate_weight_policy(weight_policy_source, weight_policy_dest)
        k = "{}-{}".format(e['channel_id'], e['node2_pub'])
        attr['policy_source'] = e["node2_policy"]
        attr['policy_dest'] = e["node1_policy"]
        attr['fee_source'] = weight_policy_source
        attr['fee_dest'] = weight_policy_dest
        attr['fee'] = weight_policy

        g2_unrestricted.add_edge(e['node2_pub'], e['node1_pub'], key=k,
                                 **attr)  # TODO:(now we can use mappings between graphs, but this may be faster)

        attrib, links_node2_fee, _, _ = set_similar_links(e, links_node2_fee, attr, True, None, None, prop='fee')
        if attrib['fee'] is not None:
            gs2_unrestricted.add_edge(e['node2_pub'], e['node1_pub'],
                                      key="{}-{}".format(e['node2_pub'], e['node1_pub']),
                                      **attrib)

        attrib, links_node2_cap, _, _ = set_similar_links(e, links_node2_cap, attr, True, None, None, prop='capacity')
        if attrib['capacity'] is not None:
            attrib['flow'] = 0
            gs1_unrestricted.add_edge(e['node2_pub'], e['node1_pub'], key="{}-{}".format(e['node2_pub'], e['node1_pub']),
                                      **attrib)

    print("TOTAL EDGES:{}".format(len(data['edges'])))
    count = 0
    num_edges = 0
    biggest_num_channels = 0
    nodes = ''
    links_nodes = {}
    for link in links:
        reverse_link = link.split('-')[1] + '-' + link.split('-')[0]
        initial_length = len(links[link]["channels"])
        length = len(links[link]["channels"]) + 0 if reverse_link not in links else len(links[reverse_link]["channels"])
        # for pubkey in ids:
        for i, pubkey in enumerate(link.split('-')):
            if pubkey in links_nodes:
                links_nodes[pubkey]['total_channels'] += length
                links_nodes[pubkey]['total_channels_disabled'] += links[link]["disabled"]
                links_nodes[pubkey]['initial_channels'] += initial_length if i == 0 else 0
                links_nodes[pubkey]['initial_channels_disabled'] += links[link]["disabled"] if i == 0 else 0
                links_nodes[pubkey]['max_channels'] = length if length > links_nodes[pubkey]['max_channels'] else \
                    links_nodes[pubkey]['max_channels']
                links_nodes[pubkey]['max_channels_disabled'] = links[link]["disabled"] \
                    if length > links_nodes[pubkey]['max_channels'] else links_nodes[pubkey]['max_channels_disabled']
            else:
                if pubkey in links_pairs:
                    links_nodes[pubkey] = {'total_channels': length, 'total_channels_disabled': links[link]["disabled"],
                                           'initial_channels': initial_length if i == 0 else 0,
                                           'initial_channels_disabled': links[link]["disabled"] if i == 0 else 0,
                                           'max_channels': length, 'max_channels_disabled': links[link]["disabled"],
                                           'channels': links_pairs[pubkey]}
        if length > 1:
            count += 1
            num_edges += length
            if length > biggest_num_channels:
                biggest_num_channels = length
                nodes = link

    pubkey_max_total_edges = max(links_nodes, key=lambda v: links_nodes[v]['total_channels'])
    pubkey_max_initial_edges = max(links_nodes, key=lambda v: links_nodes[v]['initial_channels'])
    print("PAIR OF NODES WITH MORE THAN ONE CHANNEL:{}\nNUMBER OF CHANNELS OF ALL PAIR OF NODES:{}"
          .format(count, num_edges))
    num_disabled_channels = links[nodes]['disabled']
    nodes = nodes.split('-')
    print("NODES WITH BIGGEST NUMBER OF CHANNELS:\n\t{}\n\t{}\n\t\tNUMBER OF CHANNELS:{}"
          "\n\t\tNUMBER OF DISABLED CHANNELS:{}"
          .format(nodes[0], nodes[1], biggest_num_channels, num_disabled_channels))
    extra_nodes.append(nodes[0])
    extra_nodes.append(nodes[1])
    print("NODE WITH MAX TOTAL CHANNELS:\n\t{}\n\t\tNUMBER OF CHANNELS TOTAL:{}\n\t\tDISABLED CHANNELS TOTAL:{}"
          "\n\t\tNUMBER OF CHANNELS INITIAL:{}\n\t\tDISABLED CHANNELS INITIAL:{}"
          .format(pubkey_max_total_edges, links_nodes[pubkey_max_total_edges]["total_channels"],
                  links_nodes[pubkey_max_total_edges]["total_channels_disabled"],
                  links_nodes[pubkey_max_total_edges]["initial_channels"],
                  links_nodes[pubkey_max_total_edges]["initial_channels_disabled"]))
    extra_nodes.append(pubkey_max_total_edges)
    print("NODE WITH MAX INITIAL CHANNELS:\n\t{}\n\t\tNUMBER OF CHANNELS INITIAL:{}\n\t\tDISABLED CHANNELS INITIAL:{}"
          "\n\t\tNUMBER OF CHANNELS TOTAL:{}\n\t\tDISABLED CHANNELS TOTAL:{}"
          .format(pubkey_max_initial_edges, links_nodes[pubkey_max_initial_edges]["initial_channels"],
                  links_nodes[pubkey_max_initial_edges]["initial_channels_disabled"],
                  links_nodes[pubkey_max_initial_edges]["total_channels"],
                  links_nodes[pubkey_max_initial_edges]["total_channels_disabled"]))
    extra_nodes.append(pubkey_max_initial_edges)
    for e in g2_unrestricted.edges(data=True, keys=True):
        edge_dict[e[2]] = e

    dict_graphs = {'g1_restricted': g1_restricted,
                   'g2_restricted': g2_restricted,
                   'gs2_unrestricted': gs2_unrestricted,
                   'gs2_restricted': gs2_restricted,
                   'gs1_unrestricted': gs1_unrestricted,
                   'gs1_restricted': gs1_restricted,
                   'g2_restricted_path': g2_restricted_path}

    return g1_unrestricted, g2_unrestricted, dict_graphs, node_dict, edge_dict, links_nodes, extra_nodes


def get_graphs_restrictions(g2_unrestricted: nx, dict_graphs: dict, parameters: dict) -> dict:
    g2_restricted = dict_graphs['g2_restricted']
    gs2_unrestricted = dict_graphs['gs2_unrestricted']
    gs2_restricted = dict_graphs['gs2_restricted']
    gs1_unrestricted = dict_graphs['gs1_unrestricted']
    gs1_restricted = dict_graphs['gs1_restricted']
    g2_restricted_path = dict_graphs['g2_restricted_path']

    # all_edges = [] aif not prameters["is_length_path"] else get_nodes_longest_paths(g2_unrestricted, parameters)
    count_disabled = Counter()
    count_balance = Counter()
    count_min_htlc = Counter()
    count_timeout = Counter()
    count_edges = Counter()
    count_delta = Counter()
    total_capacity_restricted_g2 = 0
    total_capacity_unrestricted_g2 = 0
    for edge1 in g2_unrestricted.edges(data=True, keys=True):
        count_edges.preinc()
        total_capacity_unrestricted_g2 += edge1[3]['balance']
        val_timeouts = sum(edge1[3]['pending_htlc'][i][1] for i in edge1[3]['pending_htlc'])

        # 6.1.0 Channel enabled
        bidirectional_edges = g2_unrestricted.get_edge_data(edge1[1], edge1[0])
        is_channel_enabled = False
        if bidirectional_edges is not None:
            key = "{}-{}".format(edge1[3]['channel_id'], edge1[1])
            if key in bidirectional_edges:
                edge2 = bidirectional_edges[key]
                is_channel_enabled = True if not parameters['flag_channel_enabled'] else \
                    True if (edge1[3]['policy_source'] is not None and 'disabled' in edge1[3]['policy_source'] and
                             not edge1[3]['policy_source']['disabled']) and \
                            (edge2['policy_source'] is not None and 'disabled' in edge2['policy_source'] and
                             not edge2['policy_source']['disabled']) else False

        count_disabled.preinc() if not is_channel_enabled else None
        if is_channel_enabled:
            total_capacity_restricted_g2 += edge1[3]['balance']
            g2_restricted.add_edge(edge1[0], edge1[1], key=edge1[2], **edge1[3])

            # 6.1.1 There is enough balance in all the channels to fulfill the payment
            is_enough_balance = True if not parameters['flag_enough_balance'] else \
                edge1[3]['balance'] > parameters['balance']
            count_balance.preinc() if not is_enough_balance else None
            # 6.1.3 The number of existing HTLCs in each channel is less than 14
            is_minimum_htlc = True if not parameters['flag_minimum_htlc'] else \
                len(edge1[3]['pending_htlc']) < parameters['num_pending_htlc']
            count_min_htlc.preinc() if not is_minimum_htlc else None
            # 6.1.4.1 There exists a set of timeouts for all HTLCs in the path that fulfill the conditions on the nodes
            # policies for all nodes in the path.
            is_valid_timeout = True if not parameters['flag_valid_timeout'] else edge1[3]['policy_source'] is not None \
                                                                                 and 'time_lock_delta' in edge1[3][
                                                                                     'policy_source'] \
                                                                                 and val_timeouts < \
                                                                                 edge1[3]['policy_source'][
                                                                                     'time_lock_delta']
            count_timeout.preinc() if not is_valid_timeout else None
            # 6.1.4.2 The amount of the payment is higher than the minimum (σ>min_htlc).
            is_payment_greater_delta = True if not parameters['flag_payment_greater_delta'] \
                else edge1[3]['policy_source'] is not None and \
                     parameters['amount_payment'] > int(edge1[3]['policy_source']['min_htlc'])
            count_delta.preinc() if not is_payment_greater_delta else None

            if is_minimum_htlc and is_enough_balance and is_valid_timeout and is_payment_greater_delta:
                if gs2_restricted.get_edge_data(edge1[0], edge1[1]) is None:
                    edge = gs2_unrestricted.get_edge_data(edge1[0], edge1[1])
                    if edge is not None and edge['fee'] is not None and edge['channel_id'] == edge1[3]['channel_id']:
                        edge['balance'] = edge1[3]['balance']
                        edge['pending_htlc'] = edge1[3]['pending_htlc']
                        edge.pop('key', None)
                        gs2_restricted.add_edge(edge1[0], edge1[1], key=edge1[2], **edge)

                g2_restricted_path.add_edge(edge1[0], edge1[1], key=edge1[2], **edge1[3])

                if gs1_restricted.get_edge_data(edge1[0], edge1[1]) is None:
                    edge = gs1_unrestricted.get_edge_data(edge1[0], edge1[1])
                    if edge is not None and edge['fee'] is not None and edge['channel_id'] == edge1[3]['channel_id']:
                        edge['flow'] = 0
                        edge['balance'] = edge1[3]['balance']
                        edge['pending_htlc'] = edge1[3]['pending_htlc']
                        edge.pop('key', None)
                        gs1_restricted.add_edge(edge1[0], edge1[1], key=edge1[2], **edge)

    print("TOTAL BALANCE G2: \n\tUnrestricted:{}\n\tRestricted:{}".format(total_capacity_unrestricted_g2,
                                                                          total_capacity_restricted_g2))
    print("INFO:\n\tChannels disabled:{}\n\tChannels no balance:{}\n\tChannels max htlc:{}"
          "\n\tChannels no timeout:{}\n\tChannels payment higher:{}\n\t\tTotal number of edges:{}"
          "\n\t\tTotal number of nodes:{}".
          format(count_disabled.v, count_balance.v, count_min_htlc.v, count_timeout.v, count_delta.v, count_edges.v,
                 len(g2_unrestricted.nodes)))
    dict_graphs_restricted = {'g2_restricted': g2_restricted,
                              'gs2_restricted': gs2_restricted,
                              'gs1_restricted': gs1_restricted,
                              'g2_restricted_path': g2_restricted_path,
                              'total_edges': count_edges.v}

    return dict_graphs_restricted


def set_centrality(g1_unrestricted: nx, g2_unrestricted: nx, dict_graphs: dict,
                   parameters: dict, links_nodes: dict, extra_nodes: list, ordinal=""):
    """
    Gets by each node the measures of degree, degree weighted and degree centrality

    :param parameters: parameters that contain tuning parameter to calculate centrality and directory to save metrics file
    :param g1_unrestricted: the multi graph network g1 unrestricted & undirected
    :param g2_unrestricted: the multi graph network g2 unrestricted & directed
    :param dict_graphs: contains the different graphs for calculating metrics
        g1_restricted: the multi graph network g1 restricted & undirected
        g2_restricted: the multi graph network g2 restricted & directed
        gs1_undirected: the graph network gs1 (undirected)
        gs1_directed: the graph network gs1 (directed)
        gs2_unrestricted: the graph network gs2 unrestricted (directed)
        gs2_restricted: the graph network gs2 restricted (directed)
        g2_unrestricted_path: the graph network g2 with path restrictions (directed)
        g2_restricted_path: the graph network g2 with path restrictions (directed)
        total_edges: total number of edges
    :param links_nodes: contains information about total channels, total channels disabled, max channels and max channels
        disabled by each node
    :param extra_nodes: Specific nodes used to calculate flow_betweenness
    :param ordinal: Number of the simulation to run
    :return: None
    """
    logging.debug("Computing centrality metrics")
    if 'filename_snapshot' in parameters:
        message = {'filename': parameters['filename_snapshot'],
                   'datetime': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                   'weight': parameters['weight'],
                   'start_metrics': datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}

    k = int(len(g1_unrestricted.nodes()) / parameters["k_fraction"]) if len(g1_unrestricted.nodes()) > 1000 else None

    g1_restricted = dict_graphs['g1_restricted']
    gs1_unrestricted = dict_graphs['gs1_unrestricted']
    gs2_unrestricted = dict_graphs['gs2_unrestricted']

    dict_graphs_restricted = get_graphs_restrictions(g2_unrestricted.copy(), dict_graphs, parameters)
    gs2_restricted = dict_graphs_restricted['gs2_restricted']
    gs1_restricted = dict_graphs_restricted['gs1_restricted']
    g2_restricted_path = dict_graphs_restricted['g2_restricted_path']
    g2_restricted = dict_graphs_restricted['g2_restricted']
    total_edges = dict_graphs_restricted['total_edges']

    # ***** DEGREE FOR UNRESTRICTED AND RESTRICTED GRAPHS WITH AND WITHOUT WEIGHT
    all_degree_unrestricted = g1_unrestricted.degree()
    all_degree_restricted = g1_restricted.degree()  # G1 weight=None
    all_strength_unrestricted = g1_unrestricted.degree(weight='capacity')  # G1 weight=capacity
    all_strength_restricted = g1_restricted.degree(weight='capacity')  # G1 weight=capacity

    extra_nodes.append(max(dict(all_degree_restricted), key=dict(all_degree_restricted).get))
    extra_nodes.append(max(dict(all_strength_restricted), key=dict(all_strength_restricted).get))
    logging.debug("Degree for unrestricted/restricted graph with/out weight computed")

    # ***** IN DEGREE FOR UNRESTRICTED AND RESTRICTED GRAPHS WITH AND WITHOUT WEIGHT
    all_incoming_degree_unrestricted = g2_unrestricted.in_degree()  # G2 weight=None
    all_incoming_strength_unrestricted = g2_unrestricted.in_degree(weight='balance')  # G2 weight=balance
    all_incoming_degree_restricted = g2_restricted.in_degree()  # G2 weight=None
    all_incoming_strength_restricted = g2_restricted.in_degree(weight='balance')  # G2	weight=balance

    extra_nodes.append(max(dict(all_incoming_degree_restricted), key=dict(all_incoming_degree_restricted).get))
    extra_nodes.append(max(dict(all_incoming_strength_restricted), key=dict(all_incoming_strength_restricted).get))
    logging.debug("In degree for unrestricted/restricted graphs with/out weight computed")

    # ***** OUT DEGREE FOR UNRESTRICTED AND RESTRICTED GRAPHS WITH AND WITHOUT WEIGHT
    all_outgoing_degree_unrestricted = g2_unrestricted.out_degree()  # G2 weight=None
    all_outgoing_strength_unrestricted = g2_unrestricted.out_degree(weight='balance')  # G2 weight=balance
    all_outgoing_degree_restricted = g2_restricted.out_degree()  # G2	weight=None
    all_outgoing_strength_restricted = g2_restricted.out_degree(weight='balance')  # G2	weight=balance

    extra_nodes.append(max(dict(all_outgoing_degree_restricted), key=dict(all_outgoing_degree_restricted).get))
    extra_nodes.append(max(dict(all_outgoing_strength_restricted), key=dict(all_outgoing_strength_restricted).get))
    logging.debug("Out degree for unrestricted/restricted graphs with/out weight computed")

    # ***** BETWEENNESS FOR UNRESTRICTED AND RESTRICTED GRAPHS WITH AND WITHOUT WEIGHT
    all_betw_unrestricted, all_betw_unrestricted_norm, all_betw_restricted, all_betw_restricted_norm, \
    all_weighted_betw_unrestricted, all_weighted_betw_unrestricted_norm, \
    all_weighted_betw_restricted, all_weighted_betw_restricted_norm, \
    all_weighted_betw_unrestricted_cap, all_weighted_betw_unrestricted_cap_norm, \
    all_weighted_betw_restricted_cap, all_weighted_betw_restricted_cap_norm = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    if parameters['flag_betweenness']:
        all_betw_unrestricted, all_betw_unrestricted_norm = betweenness_centrality(gs2_unrestricted, k=k)  # G2 weight=None
        # 6.1.2 The length of the path is smaller or equal than 20 -> parameters['length_path']
        all_betw_restricted, all_betw_restricted_norm = betweenness_centrality(
            gs2_restricted, k=k, cutoff=None if not parameters['flag_length_path']
            else parameters['length_path'])  # G2 weight=None
        all_weighted_betw_unrestricted, all_weighted_betw_unrestricted_norm = betweenness_centrality(
            gs2_unrestricted, k=k, weight='fee', cutoff=total_edges / 2)  # G2 weight='fee'
        all_weighted_betw_restricted, all_weighted_betw_restricted_norm = betweenness_centrality(
            gs2_restricted, k=k, weight='fee', cutoff=None if not parameters['flag_length_path']
            else parameters['length_path'])  # G2 weight='fee'
        all_weighted_betw_unrestricted_cap, all_weighted_betw_unrestricted_cap_norm = betweenness_centrality(
            gs1_unrestricted, k=k, weight='capacity')  # G1 weight='capacity'
        all_weighted_betw_restricted_cap, all_weighted_betw_restricted_cap_norm = betweenness_centrality(
            gs1_restricted, k=k, weight='capacity', cutoff=None if not parameters['flag_length_path']
            else parameters['length_path'])  # G1 weight='capacity'

        extra_nodes.append(max(dict(all_betw_restricted), key=dict(all_betw_restricted).get))
        extra_nodes.append(max(dict(all_weighted_betw_restricted), key=dict(all_weighted_betw_restricted).get))
        extra_nodes.append(max(dict(all_weighted_betw_restricted_cap), key=dict(all_weighted_betw_restricted_cap).get))
    logging.debug("Betweenness for unrestricted/restricted graphs with/out weight computed")

    # ***** CURRENT FLOW BETWEENNESS FOR UNRESTRICTED AND RESTRICTED GRAPHS WITH WEIGHT
    all_current_flow_unrestricted, all_current_flow_unrestricted_norm, \
    all_current_flow_restricted, all_current_flow_restricted_norm = {}, {}, {}, {}

    g_unrestricted_undirected = gs1_unrestricted.to_undirected()
    list_components = list(g_unrestricted_undirected.subgraph(c)
                           for c in nx.connected_components(g_unrestricted_undirected))
    components_nodes = [list(val.nodes()) for i, val in enumerate(list_components)]
    values_nodes_components = {}
    capacities = [0] * len(components_nodes)
    fees = [0] * len(components_nodes)
    for i, components in enumerate(list_components):
        for e in components.edges(data=True):
            capacities[i] += int(e[2]['capacity'])
            fees[i] += int(1 if e[2]['fee'] is None else e[2]['fee'])
        capacities[i] = int(capacities[i] / len(components))
        fees[i] = int(fees[i] / len(components))
        values_nodes_components[i] = {"capacity": capacities[i], 'fee': fees[i]}
    component_unrestricted = max(list_components, key=len)
    main_connected_component = component_unrestricted

    if parameters['flag_current_flow_betweenness']:
        if component_unrestricted is not None:
            all_current_flow_unrestricted, all_current_flow_unrestricted_norm = current_flow_betweenness_centrality(
                component_unrestricted, weight='capacity')  # G2 weight='capacity'

        g_restricted_undirected = gs1_restricted.to_undirected()
        component_restricted = max(list(g_restricted_undirected.subgraph(c)
                                   for c in nx.connected_components(g_restricted_undirected)), key=len)
        if component_restricted is not None and len(component_restricted) > 1:
            all_current_flow_restricted, all_current_flow_restricted_norm = current_flow_betweenness_centrality(
                component_restricted, weight='capacity')  # G2 weight='capacity'

        extra_nodes.append(max(dict(all_current_flow_restricted), key=dict(all_current_flow_restricted).get))
    logging.debug("Current flow betweenness for unrestricted/restricted graphs with weight computed")

    # ***** CLOSENESS FOR UNRESTRICTED AND RESTRICTED GRAPHS WITH AND WITHOUT WEIGHT
    all_closeness_unrestricted, all_closeness_unrestricted_norm, \
    all_closeness_restricted, all_closeness_restricted_norm,\
    all_weighted_closeness_unrestricted, all_weighted_closeness_unrestricted_norm, \
    all_weighted_closeness_restricted, all_weighted_closeness_restricted_norm = {}, {}, {}, {}, {}, {}, {}, {}
    if parameters['flag_closeness']:
        all_closeness_unrestricted, all_closeness_unrestricted_norm = closeness_centrality(
            g2_unrestricted, wf_improved=True)  # G2 weight=None
        all_closeness_restricted, all_closeness_restricted_norm = closeness_centrality(
            g2_restricted_path, cutoff=None if not parameters['flag_length_path']
            else parameters['length_path'], wf_improved=True)  # G2 weight=None
        all_weighted_closeness_unrestricted, all_weighted_closeness_unrestricted_norm = closeness_centrality(
            g2_unrestricted, distance='fee', cutoff=None, wf_improved=True)  # G2 weight='fee'
        all_weighted_closeness_restricted, all_weighted_closeness_restricted_norm = closeness_centrality(
            g2_restricted_path, distance='fee', cutoff=None if not parameters['flag_length_path']
            else parameters['length_path'], wf_improved=True)  # G2 weight='fee'

        extra_nodes.append(max(dict(all_closeness_restricted), key=dict(all_closeness_restricted).get))
        extra_nodes.append(max(dict(all_weighted_closeness_restricted), key=dict(all_weighted_closeness_restricted).get))
    logging.debug("Closeness for unrestricted and restricted graphs with and without weight computed")

    # ***** FLOW BETWEENNESS FOR UNRESTRICTED AND RESTRICTED GRAPHS WITH WEIGHT
    all_weighted_flow_unrestricted, all_weighted_flow_unrestricted_norm, \
    all_weighted_flow_restricted, all_weighted_flow_restricted_norm = {}, {}, {}, {}

    if parameters['flag_flow_betweenness']:
        extra_nodes = list(dict.fromkeys(extra_nodes))
        print('Length:', len(extra_nodes), ' Nodes:', extra_nodes)
        k = parameters['num_nodes_max'] - len(extra_nodes) # int(k/3.15) if len(gs1_unrestricted.nodes()) > 1890 else k

        all_weighted_flow_unrestricted, all_weighted_flow_unrestricted_norm = flow_betweenness_centrality(
            gs1_unrestricted.copy(), weight='capacity', k=k, cutoff=None if not parameters['flag_length_path']
            else parameters['length_path'], extra_nodes=extra_nodes, debug=None)

        all_weighted_flow_restricted, all_weighted_flow_restricted_norm = flow_betweenness_centrality(
            gs1_restricted.copy(), weight='capacity', k=k, cutoff=None if not parameters['flag_length_path']
            else parameters['length_path'], extra_nodes=extra_nodes)
    logging.debug("Flow betweenness for unrestricted/restricted graphs with weight computed")

    # ***** PAGERANK FOR UNRESTRICTED AND RESTRICTED GRAPHS WITH AND WITHOUT WEIGHT
    all_pr_unrestricted = nx.pagerank(gs2_restricted)  # G2 weight='None'
    all_pr_restricted = nx.pagerank(gs2_restricted)  # G2 weight='None'
    all_weighted_pr_unrestricted = nx.pagerank(gs2_unrestricted, weight='fee')  # G2 weight='fee'
    all_weighted_pr_restricted = nx.pagerank(gs2_unrestricted, weight='fee')  # G2 weight='fee'
    logging.debug("Pagerank for unrestricted and restricted graphs with and without weight computed")

    alpha_strength = parameters["alpha_strength"]
    flag_flow_betweenness = parameters['flag_flow_betweenness']
    flag_current_flow_betweenness = parameters['flag_current_flow_betweenness']

    for node, degree_unrestricted in all_degree_unrestricted:  # G1 weight=None
        degree_restricted = all_degree_restricted[node]
        strength_unrestricted = all_strength_unrestricted[node]
        strength_restricted = all_strength_restricted[node]
        incoming_strength_unrestricted = all_incoming_strength_unrestricted[node]
        incoming_strength_restricted = all_incoming_strength_restricted[node]
        outgoing_strength_unrestricted = all_outgoing_strength_unrestricted[node]
        outgoing_strength_restricted = all_outgoing_strength_restricted[node]
        # Node centrality in weighted networks: Generalizing degree and shortest paths - G1 weight='capacity'
        opsahl_unrestricted = round(pow(degree_unrestricted, (1 - parameters["alpha_centrality"]))
                                    * pow(strength_unrestricted, parameters["alpha_centrality"]), 4)
        opsahl_restricted = round(pow(degree_restricted, (1 - parameters["alpha_centrality"]))
                                  * pow(strength_restricted, parameters["alpha_centrality"]), 4)
        # Degree Centrality and Variation in Tie Weights - G2 weight='balance'
        incoming_opsahl_unrestricted = get_incoming_outgoing_opsahl(all_incoming_degree_unrestricted[node],
                                                                    all_incoming_strength_unrestricted[node],
                                                                    alpha_strength)
        incoming_opsahl_restricted = get_incoming_outgoing_opsahl(all_incoming_degree_restricted[node],
                                                                  all_incoming_strength_restricted[node],
                                                                  alpha_strength)
        outgoing_opsahl_unrestricted = get_incoming_outgoing_opsahl(all_outgoing_degree_unrestricted[node],
                                                                    all_outgoing_strength_unrestricted[node],
                                                                    alpha_strength)
        outgoing_opsahl_restricted = get_incoming_outgoing_opsahl(all_outgoing_degree_restricted[node],
                                                                  all_outgoing_strength_restricted[node],
                                                                  alpha_strength)
        betweenness_unrestricted = get_betweenness(node, all_betw_unrestricted)
        betweenness_unrestricted_norm = get_betweenness(node, all_betw_unrestricted_norm)
        betweenness_restricted = get_betweenness(node, all_betw_restricted)
        betweenness_restricted_norm = get_betweenness(node, all_betw_restricted_norm)
        weighted_betweenness_unrestricted = get_betweenness(node, all_weighted_betw_unrestricted)
        weighted_betweenness_unrestricted_norm = get_betweenness(node, all_weighted_betw_unrestricted_norm)
        weighted_betweenness_restricted = get_betweenness(node, all_weighted_betw_restricted)
        weighted_betweenness_restricted_norm = get_betweenness(node, all_weighted_betw_restricted_norm)

        weighted_betweenness_unrestricted_cap = get_betweenness(node, all_weighted_betw_unrestricted_cap)
        weighted_betweenness_unrestricted_cap_norm = get_betweenness(node, all_weighted_betw_unrestricted_cap_norm)
        weighted_betweenness_restricted_cap = get_betweenness(node, all_weighted_betw_restricted_cap)
        weighted_betweenness_restricted_cap_norm = get_betweenness(node, all_weighted_betw_restricted_cap_norm)

        flow_based_betweenness_unrestricted = get_flow(node, all_weighted_flow_unrestricted, flag_flow_betweenness)
        flow_based_betweenness_unrestricted_norm = get_flow(node, all_weighted_flow_unrestricted_norm,
                                                            flag_flow_betweenness)
        flow_based_betweenness_restricted = get_flow(node, all_weighted_flow_restricted, flag_flow_betweenness)
        flow_based_betweenness_restricted_norm = get_flow(node, all_weighted_flow_restricted_norm,
                                                          flag_flow_betweenness)

        current_flow_based_betweenness_unrestricted = get_flow(node, all_current_flow_unrestricted,
                                                               flag_current_flow_betweenness)
        current_flow_based_betweenness_unrestricted_norm = get_flow(node, all_current_flow_unrestricted_norm,
                                                                    flag_current_flow_betweenness)
        current_flow_based_betweenness_restricted = get_flow(node, all_current_flow_restricted,
                                                             flag_current_flow_betweenness)
        current_flow_based_betweenness_restricted_norm = get_flow(node, all_current_flow_restricted_norm,
                                                                  flag_current_flow_betweenness)

        closeness_unrestricted = get_closeness(node, all_closeness_unrestricted)
        closeness_unrestricted_norm = get_closeness(node, all_closeness_unrestricted_norm)
        closeness_restricted = get_closeness(node, all_closeness_restricted)
        closeness_restricted_norm = get_closeness(node, all_closeness_restricted_norm)
        weighted_closeness_unrestricted = get_closeness(node, all_weighted_closeness_unrestricted)
        weighted_closeness_unrestricted_norm = get_closeness(node, all_weighted_closeness_unrestricted_norm)
        weighted_closeness_restricted = get_closeness(node, all_weighted_closeness_restricted)
        weighted_closeness_restricted_norm = get_closeness(node, all_weighted_closeness_restricted_norm)

        pagerank_unrestricted = get_pagerank(node, all_pr_unrestricted)
        pagerank_restricted = get_pagerank(node, all_pr_restricted)
        weighted_pagerank_unrestricted = get_pagerank(node, all_weighted_pr_unrestricted)
        weighted_pagerank_restricted = get_pagerank(node, all_weighted_pr_restricted)

        component = [(i, nodes.index(node)) for i, nodes in enumerate(components_nodes) if node in nodes][0][0]

        metrics = {'capacity': values_nodes_components[component]['capacity'],
                   'fee': values_nodes_components[component]['fee'],
                   'main_connected_component': True if node in main_connected_component else False,
                   'connected_component': component,
                   'degree_unrestricted': degree_unrestricted, 'degree_restricted': degree_restricted,
                   'strength_unrestricted': strength_unrestricted, 'strength_restricted': strength_restricted,
                   'incoming_strength_unrestricted': incoming_strength_unrestricted,
                   'incoming_strength_restricted': incoming_strength_restricted,
                   'outgoing_strength_unrestricted': outgoing_strength_unrestricted,
                   'outgoing_strength_restricted': outgoing_strength_restricted,
                   'opsahl_unrestricted': opsahl_unrestricted, 'opsahl_restricted': opsahl_restricted,
                   'incoming_opsahl_unrestricted': incoming_opsahl_unrestricted,
                   'incoming_opsahl_restricted': incoming_opsahl_restricted,
                   'outgoing_opsahl_unrestricted': outgoing_opsahl_unrestricted,
                   'outgoing_opsahl_restricted': outgoing_opsahl_restricted,
                   'betweenness_unrestricted': betweenness_unrestricted,
                   'betweenness_unrestricted_norm': betweenness_unrestricted_norm,
                   'betweenness_restricted': betweenness_restricted,
                   'betweenness_restricted_norm': betweenness_restricted_norm,
                   'weighted_betweenness_unrestricted': weighted_betweenness_unrestricted,
                   'weighted_betweenness_unrestricted_norm': weighted_betweenness_unrestricted_norm,
                   'weighted_betweenness_restricted': weighted_betweenness_restricted,
                   'weighted_betweenness_restricted_norm': weighted_betweenness_restricted_norm,
                   'weighted_betweenness_cap_unrestricted': weighted_betweenness_unrestricted_cap,
                   'weighted_betweenness_cap_unrestricted_norm': weighted_betweenness_unrestricted_cap_norm,
                   'weighted_betweenness_cap_restricted': weighted_betweenness_restricted_cap,
                   'weighted_betweenness_cap_restricted_norm': weighted_betweenness_restricted_cap_norm,
                   'flow_based_betweenness_unrestricted': flow_based_betweenness_unrestricted,
                   'flow_based_betweenness_unrestricted_norm': flow_based_betweenness_unrestricted_norm,
                   'flow_based_betweenness_restricted': flow_based_betweenness_restricted,
                   'flow_based_betweenness_restricted_norm': flow_based_betweenness_restricted_norm,
                   'current_flow_based_betweenness_unrestricted': current_flow_based_betweenness_unrestricted,
                   'current_flow_based_betweenness_unrestricted_norm': current_flow_based_betweenness_unrestricted_norm,
                   'current_flow_based_betweenness_restricted': current_flow_based_betweenness_restricted,
                   'current_flow_based_betweenness_restricted_norm': current_flow_based_betweenness_restricted_norm,
                   'closeness_unrestricted': closeness_unrestricted,
                   'closeness_unrestricted_norm': closeness_unrestricted_norm,
                   'closeness_restricted': closeness_restricted,
                   'closeness_restricted_norm': closeness_restricted_norm,
                   'weighted_closeness_unrestricted': weighted_closeness_unrestricted,
                   'weighted_closeness_unrestricted_norm': weighted_closeness_unrestricted_norm,
                   'weighted_closeness_restricted': weighted_closeness_restricted,
                   'weighted_closeness_restricted_norm': weighted_closeness_restricted_norm,
                   'pagerank_unrestricted': pagerank_unrestricted, 'pagerank_restricted': pagerank_restricted,
                   'weighted_pagerank_unrestricted': weighted_pagerank_unrestricted,
                   'weighted_pagerank_restricted': weighted_pagerank_restricted,
                   'total_channels': 0 if node not in links_nodes else links_nodes[node]['total_channels'],
                   'total_channels_disabled': 0 if node not in links_nodes else
                   links_nodes[node]['total_channels_disabled'],
                   'max_channels': 0 if node not in links_nodes else links_nodes[node]['max_channels'],
                   'max_channels_disabled': 0 if node not in links_nodes else
                   links_nodes[node]['max_channels_disabled'],
                   "channels": None if node not in links_nodes else links_nodes[node]['channels']
                   }
        attrs = g1_unrestricted.nodes[node]
        attrs.update(metrics)
        node_attributes = {node: attrs}
        nx.set_node_attributes(g1_unrestricted, node_attributes)

        attrs = g2_unrestricted.nodes[node]
        attrs.update(metrics)
        node_attributes = {node: attrs}
        nx.set_node_attributes(g2_unrestricted, node_attributes)

    logging.debug("Saving graphs into pickle files")
    save_pickle(parameters['location_results'], parameters['filename_snapshot'].split('.')[0] + '_G1' + ordinal + '.pickle',
                g1_unrestricted)
    save_pickle(parameters['location_results'], parameters['filename_snapshot'].split('.')[0] + '_G2' + ordinal + '.pickle',
                g2_unrestricted)


def get_incoming_outgoing_opsahl(degree, strength, alpha_strength):
    return 0 if degree == 0 else round(degree * pow(strength / degree, alpha_strength), 4)


def get_betweenness(node, all_weighted_betweenness):
    return round(0.0 if node not in all_weighted_betweenness or math.isnan(all_weighted_betweenness[node])
                 else all_weighted_betweenness[node], 4)


def get_flow(node, all_flow_betweenness, flag):
    return round(all_flow_betweenness[node], 4) if node in all_flow_betweenness else round(0, 4) if flag else 0.0


def get_closeness(node, all_closeness):
    return round(0.0 if node not in all_closeness or math.isnan(all_closeness[node]) \
                 else all_closeness[node], 4)


def get_pagerank(node, all_page_rank):
    return round(0.0 if node not in all_page_rank or math.isnan(all_page_rank[node]) \
                 else all_page_rank[node], 4)


def transform_to_graph(g1: nx):
    """
    Transforms a multigraph/multidigraph to a graph

    :param g1: the graph network g1 (undirected)
    :return: a graph
    """
    g = nx.Graph()
    for u, v, data in g1.edges(data=True):
        policy = data["fee_source"] if data["fee_source"] < data["fee_dest"] \
            else data["fee_dest"]

        if g.has_edge(u, v):
            g[u][v]["capacity"] += data["capacity"]
            g[u][v]["policy"] += policy
        else:

            g.add_edge(u, v, capacity=data["capacity"], policy=policy)

    return g


def replace_path_os(path: str) -> str:
    """
    Replaces the path according to the operating system

    :param path: path to replace according to OS
    :return: path replaced
    """
    if os.sys.platform == 'win32':
        return path.replace('/', '\\')

    return path


def validate_dir(path: str) -> bool:
    """
    Validates whether a path exist or not

    :param path: path to validate
    :return: bool
    """
    try:
        with open(path):  # OSError if file does not exist or is invalid
            return True
    except OSError as err:
        if os.path.exists(path):
            return True
        else:
            return False


def get_pubkey_alias(alias: str, graph: nx) -> str:
    """
    Returns the pub_key from an alias

    :param alias: Node alias
    :param graph: multigraph with the whole data about the network
    :return: pubkey
    """
    pub_key = None
    for n in graph.nodes(data=True):
        if alias == n[1]['alias']:
            pub_key = n[0]
            if pub_key is not None:
                break

    return pub_key


def get_alias_pubkey(pubkey: str, graph: nx) -> str:
    """
    Returns the pub_key from an alias

    :param pubkey: Node pubkey
    :param graph: multigraph with the whole data about the network
    :return: alias
    """
    alias = None
    for n in graph.nodes(data=True):
        if pubkey == n[0]:
            alias = n[1]['alias']
            if alias is not None:
                break

    return alias


class Counter(object):
    def __init__(self, v=0):
        self.set(v)

    def preinc(self):
        self.v += 1
        return self.v

    def predec(self):
        self.v -= 1
        return self.v

    def postinc(self):
        self.v += 1
        return self.v - 1

    def postdec(self):
        self.v -= 1
        return self.v + 1

    def __add__(self, addend):
        return self.v + addend

    def __sub__(self, subtrahend):
        return self.v - subtrahend

    def __mul__(self, multiplier):
        return self.v * multiplier

    def __div__(self, divisor):
        return self.v / divisor

    def __getitem__(self):
        return self.v

    def __str__(self):
        return str(self.v)

    def set(self, v):
        if type(v) != int:
            v = 0
        self.v = v
