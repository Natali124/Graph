import torch
import torch_geometric

from torch_geometric.data import Data
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

import numpy as np
import itertools
import random
import math

class TreeDataset(object):
    def __init__(self, depth):
        super(TreeDataset, self).__init__()
        self.depth = depth
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()
        self.criterion = F.cross_entropy

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self, add_self_loops=True):
        edge_index = torch.tensor(self.edges).t()
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, )
        return edge_index

    def generate_data(self, train_fraction):
        data_list = []

        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(add_self_loops=True)
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=torch.long)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = self.label(comb)
            data_list.append(Data(x=nodes, edge_index=edge_index, root_mask=root_mask, y=label))

        dim0, out_dim = self.get_dims()
        X_train, X_test = train_test_split(
            data_list, train_size=train_fraction, shuffle=True, stratify=[data.y for data in data_list])


        return X_train, X_test, dim0, out_dim, self.criterion

    # Every sub-class should implement the following methods:
    def get_combinations(self):
        raise NotImplementedError

    def get_nodes_features(self, combination):
        raise NotImplementedError

    def label(self, combination):
        raise NotImplementedError

    def get_dims(self):
        raise NotImplementedError

class DictionaryLookupDataset(TreeDataset):
    def __init__(self, depth):
        super(DictionaryLookupDataset, self).__init__(depth)

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!)*num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 1000
        max_examples = 32000

        if self.depth > 3:
            per_depth_num_permutations = min(num_permutations, math.factorial(num_leaves), max_examples // num_leaves)
            permutations = [np.random.permutation(range(1, num_leaves + 1)) for _ in
                            range(per_depth_num_permutations)]
        else:
            permutations = random.sample(list(itertools.permutations(range(1, num_leaves + 1))),
                                         min(num_permutations, math.factorial(num_leaves)))

        return itertools.chain.from_iterable(

            zip(range(1, num_leaves + 1), itertools.repeat(perm))
            for perm in permutations)

    def get_nodes_features(self, combination):
        # combination: a list of indices
        # Each leaf contains a one-hot encoding of a key, and a one-hot encoding of the value
        # Every other node is empty, for now
        selected_key, values = combination

        # selected key ranges from 1 to 2^Nnodes, permutation is list of 1 to 2^Nnodes

        # The root is [one-hot selected key] + [0 ... 0]
        nodes = [ (selected_key, 0) ]

        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i) # leaf index
                node = (leaf_num+1, values[leaf_num]) # leaf index, permutation
            else:
                node = (0, 0)
            nodes.append(node)
        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key - 1])

    def get_dims(self):
        # get input and output dims
        in_dim = len(self.leaf_indices)
        out_dim = len(self.leaf_indices)
        return in_dim, out_dim




def create_graph_with_cliques_and_central_node():
    """
    Creates a clustered graph
    """
    num_cliques = 4
    nodes_per_clique = 5
    central_node = 0  # Index of the central node
    
    edge_index = []
    current_node = 1  # Start numbering nodes after the central node
    
    # Random features for each node
    node_features = []
    node_features.append([random.random() for _ in range(3)])

    # Create cliques and connect them to the central node
    for _ in range(num_cliques):
        clique_nodes = list(range(current_node, current_node + nodes_per_clique))
        
        # Add random features for nodes in the clique
        for _ in clique_nodes:
            node_features.append([random.random() for _ in range(3)])  # 3 features per node
        
        # Add edges for the clique (complete graph within the clique)
        for u, v in itertools.combinations(clique_nodes, 2):
            edge_index.append([u, v])
            edge_index.append([v, u])  # For undirected graph
        
        # Connect one node from the clique to the central node
        edge_index.append([clique_nodes[0], central_node])
        edge_index.append([central_node, clique_nodes[0]])  # Undirected edge
        
        # Move to the next set of nodes for the next clique
        current_node += nodes_per_clique

    # Format it
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    num_nodes = current_node  # Total number of nodes (central + all clique nodes)
    data = Data(edge_index=edge_index, x=x, num_nodes=num_nodes)

    return data