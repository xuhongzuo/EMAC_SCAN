import numpy as np
import random


class Graph:
    def __init__(self, nx_G, adjacency_matrix):
        self.G = nx_G
        self.adjacency_matrix = adjacency_matrix
        self.neighbours = []
        self.p = 1
        self.q = 1

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        neighbours = self.neighbours
        walk = [start_node]

        while len(walk) < walk_length:
            current_node = walk[-1]
            cur_neighbours = neighbours[int(current_node)]
            if len(cur_neighbours) > 0:
                if len(walk) == 1:
                    walk.append(cur_neighbours[alias_draw(alias_nodes[current_node][0], alias_nodes[current_node][1])])
                else:
                    prev = walk[-2]
                    next = cur_neighbours[alias_draw(alias_edges[(prev, current_node)][0],
                                               alias_edges[(prev, current_node)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def preprocess_transition_probs(self):
        """
        Pre-processing of transition probabilities for guiding the random walks.
        """
        G = self.G
        adjacency_matrix = self.adjacency_matrix

        neighbours = []
        for i in range(len(self.adjacency_matrix)):
            neighbour = [str(kk) for kk, prob in enumerate(adjacency_matrix[i]) if prob != 0]
            neighbours.append(neighbour)
        self.neighbours = neighbours

        alias_nodes = {}
        for i in range(len(adjacency_matrix)):
            unnormalized_probs = [adjacency_matrix[i][k] for k in range(len(adjacency_matrix)) if adjacency_matrix[i][k] != 0]
            const = np.sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / const for u_prob in unnormalized_probs]
            alias_nodes[str(i)] = alias_setup(normalized_probs)

        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = alias_nodes[edge[1]]
            alias_edges[(edge[1], edge[0])] = alias_nodes[edge[0]]

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]