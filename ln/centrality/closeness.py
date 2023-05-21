import functools
import networkx as nx
from itertools import count
from heapq import heappush, heappop


def closeness_centrality(G, u=None, distance=None, wf_improved=True, cutoff=None):
    if G.is_directed():
        G = G.reverse()  # create a reversed graph view

    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(
            single_source_dijkstra_path_length, weight=distance
        )
    else:
        path_length = single_source_shortest_path_length

    if u is None:
        nodes = G.nodes
    else:
        nodes = [u]
    closeness = {}
    closeness_norm = {}
    for n in nodes:
        sp = path_length(G, n, cutoff=cutoff)
        totsp = sum(sp.values())
        len_G = len(G)
        _closeness_centrality = 0.0
        _closeness_centrality_norm = 0.0
        if totsp > 0.0 and len_G > 1:
            _closeness_centrality = (len(sp) - 1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if wf_improved:
                s = (len(sp) - 1.0) / (len_G - 1)
                _closeness_centrality_norm = _closeness_centrality * s
        closeness[n] = _closeness_centrality
        closeness_norm[n] = _closeness_centrality_norm
    if u is not None:
        return closeness[u], closeness_norm[u]
    else:
        return closeness, closeness_norm


'''
    PATH LENGTH WITH DISTANCE
'''


def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    return multi_source_dijkstra_path_length(G, {source}, cutoff=cutoff, weight=weight)


def multi_source_dijkstra_path_length(G, sources, cutoff=None, weight="weight"):
    if not sources:
        raise ValueError("sources must not be empty")
    weight = _weight_function(G, weight)
    return _dijkstra_multisource(G, sources, weight, cutoff=cutoff)


def _weight_function(G, weight):
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min((attr.get(weight, 1) for attr in d.values() if attr.get(weight) is not None),
                                   default=None)
    return lambda u, v, data: data.get(weight, 1)


def _dijkstra_multisource(G, sources, weight, pred=None, paths=None, cutoff=None, target=None):
    G_succ = G._succ if G.is_directed() else G._adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    length = {}
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    for source in sources:
        if source not in G:
            raise nx.NodeNotFound(f"Source {source} not in G")
        seen[source] = 0
        push(fringe, (0, next(c), 0, source))
    while fringe:
        (d, _, l, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        length[v] = l
        if v == target:
            break
        for u, e in G_succ[v].items():
            cost = weight(v, u, e)
            if cost is None:
                continue
            diameter = 0 if cost <= 1 else 1
            vu_dist = dist[v] + cost
            vu_length = length[v] + diameter
            if cutoff is not None:
                if vu_length > cutoff:
                    continue
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
                elif pred is not None and vu_dist == u_dist:
                    pred[u].append(v)
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), vu_length, u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the pred and paths objects passed as arguments.
    return dist


'''
    PATH LENGTH WITHOUT DISTANCE
'''


def single_source_shortest_path_length(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} is not in G")
    if cutoff is None:
        cutoff = float("inf")
    nextlevel = {source: 1}
    return dict(_single_shortest_path_length(G.adj, nextlevel, cutoff))


def _single_shortest_path_length(adj, firstlevel, cutoff):
    seen = {}  # level (number of hops) when seen in BFS
    level = 0  # the current level
    nextlevel = set(firstlevel)  # set of nodes to check at next level
    n = len(adj)
    while nextlevel and cutoff >= level:
        thislevel = nextlevel  # advance to next level
        nextlevel = set()  # and start a new set (fringe)
        found = []
        for v in thislevel:
            if v not in seen:
                seen[v] = level  # set the level of vertex v
                found.append(v)
                yield (v, level)
        if len(seen) == n:
            return
        for v in found:
            nextlevel.update(adj[v])
        level += 1
    del seen
