import numpy as np
import scipy.sparse
import networkx as nx
from networkx.utils import (
    not_implemented_for,
    reverse_cuthill_mckee_ordering,
)


@not_implemented_for("directed")
def current_flow_betweenness_centrality(G, weight=None, dtype=float, solver="full"):
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected.")
    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    # make a copy with integer labels according to rcm ordering
    # this could be done without a copy if we really wanted to
    mapping = dict(zip(ordering, range(n)))
    H = nx.relabel_nodes(G, mapping)
    betweenness = dict.fromkeys(H, 0.0)  # b[v]=0 for v in H
    for row, (s, t) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        pos = dict(zip(row.argsort()[::-1], range(n)))
        for i in range(n):
            betweenness[s] += (i - pos[i]) * row[i]
            betweenness[t] += (n - i - 1 - pos[i]) * row[i]

    betweenness_norm = _rescale(H, betweenness, n, normalized=True)
    betweenness = _rescale(H, betweenness, n, normalized=False)

    betweenness = {ordering[k]: v for k, v in betweenness.items()}
    betweenness_norm = {ordering[k]: v for k, v in betweenness_norm.items()}

    return betweenness, betweenness_norm


def _rescale(H, betweenness, n, normalized):
    bet = {}
    if normalized:
        nb = (n - 1.0) * (n - 2.0)  # normalization factor
    else:
        nb = 2.0
    for v in H:
        bet[v] = float((betweenness[v] - v) * 2.0 / nb)

    return bet


def flow_matrix_row(G, weight=None, dtype=float, solver="lu"):
    # Generate a row of the current-flow matrix
    solvername = {
        "full": FullInverseLaplacian,
    }
    n = G.number_of_nodes()
    L = laplacian_sparse_matrix(
        G, nodelist=range(n), weight=weight, dtype=dtype, format="csc"
    )
    C = solvername[solver](L, dtype=dtype)  # initialize solver
    w = C.w  # w is the Laplacian matrix width
    # row-by-row flow matrix
    for u, v in sorted(sorted((u, v)) for u, v in G.edges()):
        B = np.zeros(w, dtype=dtype)
        c = G[u][v].get(weight, 1.0)
        B[u % w] = c
        B[v % w] = -c
        # get only the rows needed in the inverse laplacian
        # and multiply to get the flow matrix row
        row = np.dot(B, C.get_rows(u, v))
        yield row, (u, v)


def laplacian_sparse_matrix(G, nodelist=None, weight=None, dtype=None, format="csr"):
    A = nx.to_scipy_sparse_matrix(
        G, nodelist=nodelist, weight=weight, dtype=dtype, format=format
    )
    (n, n) = A.shape
    data = np.asarray(A.sum(axis=1).T)
    D = scipy.sparse.spdiags(data, 0, n, n, format=format)
    return D - A


class InverseLaplacian:
    def __init__(self, L, width=None, dtype=None):
        global np
        import numpy as np

        (n, n) = L.shape
        self.dtype = dtype
        self.n = n
        if width is None:
            self.w = self.width(L)
        else:
            self.w = width
        self.C = np.zeros((self.w, n), dtype=dtype)
        self.L1 = L[1:, 1:]
        self.init_solver(L)

    def init_solver(self, L):
        pass

    def solve(self, r):
        raise nx.NetworkXError("Implement solver")

    def solve_inverse(self, r):
        raise nx.NetworkXError("Implement solver")

    def get_rows(self, r1, r2):
        for r in range(r1, r2 + 1):
            self.C[r % self.w, 1:] = self.solve_inverse(r)
        return self.C

    def get_row(self, r):
        self.C[r % self.w, 1:] = self.solve_inverse(r)
        return self.C[r % self.w]

    def width(self, L):
        m = 0
        for i, row in enumerate(L):
            w = 0
            x, y = np.nonzero(row)
            if len(y) > 0:
                v = y - i
                w = v.max() - v.min() + 1
                m = max(w, m)
        return m


class FullInverseLaplacian(InverseLaplacian):
    def init_solver(self, L):
        self.IL = np.zeros(L.shape, dtype=self.dtype)
        self.IL[1:, 1:] = np.linalg.inv(self.L1.todense())

    def solve(self, rhs):
        s = np.zeros(rhs.shape, dtype=self.dtype)
        s = np.dot(self.IL, rhs)
        return s

    def solve_inverse(self, r):
        return self.IL[r, 1:]
