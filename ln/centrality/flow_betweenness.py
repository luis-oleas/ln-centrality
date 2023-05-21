import math
import random
import multiprocessing as mp
from functools import partial


def flow_betweenness_centrality(G, weight='capacity', k=None, cutoff=None, normalized=True, debug=None, extra_nodes=[]):
    random.seed(k)
    min_weight = [y[2][weight] for y in sorted(G.edges(data=True), key=lambda x: x[2][weight]) if y[2][weight] > 0][0]
    rescale = 10 ** int(math.log10(min_weight))
    for edge in G.edges(data=True):
        edge[2][weight] = int(edge[2][weight] / rescale)
    ordering = G.nodes() if k is None else random.sample(G.nodes(), k)
    [ordering.append(x) for x in extra_nodes if x not in ordering]
    ordering.sort()
    nodes_flowbet = dict.fromkeys(ordering, 0)
    flow_betweenness = dict.fromkeys(G, 0.0)
    flow_betweenness_norm = dict.fromkeys(G, 0.0)

    def split_ordering(range_ordering, length_ordering):
        k, m = divmod(len(range_ordering), length_ordering)
        return (range_ordering[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(length_ordering))

    n = len(ordering)
    POOL_SIZE = mp.cpu_count()
    num_splits = 1 if n < POOL_SIZE else POOL_SIZE
    range_splits = split_ordering(range(0, n), num_splits)
    pool = mp.Pool(POOL_SIZE)
    flow_values = pool.map(partial(flow_betweenness_worker, G, nodes_flowbet, ordering, weight, k, cutoff, normalized, debug),
                           range_splits)
    pool.close()
    for flows in flow_values:
        for key, value in flows[0].items():
            flow_betweenness[key] += value / num_splits
        for key, value in flows[1].items():
            flow_betweenness_norm[key] += value / num_splits

    return flow_betweenness, flow_betweenness_norm


def flow_betweenness_worker(G, nodes_flowbet, ordering, weight, k, cutoff, normalized, debug, ranges):
    visited_nodes = []
    for node in ordering[0:len(ordering)]:
        cum_mflow, cum_fnode, index = 0, 0, 0
        for source in ordering[ranges.start: ranges.stop - 1]:
            for sink in ordering[ranges.start + 1:ranges.stop]:
                mflow, fnode = 0, 0
                if source != node != sink and source != sink and (source, sink) not in visited_nodes:
                    index += 1
                    mflow, fnode = ford_fulkerson(G.copy(), source, sink, node, weight, cutoff, debug, index)
                    visited_nodes.append((sink, source))
                cum_mflow += mflow
                cum_fnode += fnode
        nodes_flowbet[node] = 0.0 if cum_mflow == 0 else cum_fnode / cum_mflow
        # print("node:{}, value:{}".format(node, nodes_flowbet[node]))

    nodes_flowbet_norm = _rescale(nodes_flowbet, len(G.nodes()), normalized, directed=G.is_directed(), k=k)

    return nodes_flowbet, nodes_flowbet_norm


def ford_fulkerson(graph, source, sink, bet, weight, cutoff, debug=None, index=0):
    flow, path, flow_node = 0, True, 0
    count = 0
    while path:
        # search for path with flow reserve
        path, reserve = depth_first_search(graph, source, sink, weight, cutoff)
        count += 1
        # increase flow along the path

        if len(path) > 1 and path[0] == source and path[-1] == sink:
            flow += reserve
            for v, u in zip(path, path[1:]):
                if graph.has_edge(v, u):
                    graph[v][u]['flow'] += reserve
                    if bet == u:
                        flow_node += reserve
                else:
                    graph[u][v]['flow'] -= reserve

        # show intermediate results
        if callable(debug):
            debug(source, sink, bet, path, reserve, flow, flow_node, count, index)

    return flow, flow_node


def depth_first_search(graph, source, sink, weight, cutoff):
    # undirected = graph.to_undirected()
    explored = {source}
    stack = [(source, 0, dict(graph[source]))]

    while stack:
        v, _, neighbours = stack[-1]
        if v == sink:
            break
        if cutoff is not None and len(stack) == cutoff and stack[-1][0] != sink:
            stack.pop()
            v, _, neighbours = stack[-1]

        # search the next neighbour
        while neighbours:
            u, e = neighbours.popitem()
            if u not in explored:
                break
        else:
            stack.pop()
            continue

        # current flow and capacity
        in_direction = graph.has_edge(v, u)
        capacity = e[weight]
        flow = e['flow']
        neighbours = dict(graph[u])

        # increase or redirect flow at the edge
        if in_direction and flow < capacity:
            stack.append((u, capacity - flow, neighbours))
            explored.add(u)
        elif not in_direction and flow:
            stack.append((u, flow, neighbours))
            explored.add(u)

    # (source, sink) path and its flow reserve
    reserve = min((f for _, f, _ in stack[1:]), default=0)
    path = [v for v, _, _ in stack]

    return path, reserve


def flow_debug(i, j, v, path, reserve, flow, fnode, count_paths, index):
    if len(path) == 0:
        print('#', index, 'nodes: v=', v, ' i=', i, ' j=', j,
              'flow increased by', reserve,
              # 'at path', path,
              '; max flow:', flow, ' flow node:', fnode, '; # paths:', count_paths)


def _rescale(betweenness, n, normalized, directed=False, k=None, endpoints=False):
    betweenness_norm = {}
    if normalized:
        if endpoints:
            if n < 2:
                scale = None  # no normalization
            else:
                # Scale factor should include endpoint nodes
                scale = 1 / (n * (n - 1))
        elif n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness.items():
            betweenness_norm[v[0]] = v[1] * scale

    return betweenness_norm