# # **Feynman diagram dataset builder**
#
# Original file is located at
#     https://colab.research.google.com/drive/1Zt6BV0pZzdkWU8Pq2LDjvN4p72Wmg15n


## Standard libraries
import itertools

from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

## Imports for plotting
from IPython.display import set_matplotlib_formats


from typeguard import typechecked

set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0

DATASETPATH = "./data"
raw_filepath = f"{DATASETPATH}/raw"


# # **Plan**
#
#
# ---
#
#
# ### Nodes:
# Node list.
#
# * 1 - Internal node
# * 2 - Initial state nodes
# * 3 - Final state nodes
#
# The intention is to add information about the time direction. Could investigate the effect of including this.
#
# e.g.
#
# ```
# node_features = [2, 2, 1, 1, 3, 3]
# ```
#
#
#
# ---
#
#
# ### Edge features/attributes:
# Each edge comes with a list of features.
#
# \begin{equation}
# l = \begin{bmatrix}
# m, & S, &LI^{W}_3, & LY, &RI^W_3, &RY,  &\text{red}, &\text{blue}, &\text{green},&\text{anti-red},&\text{anti-blue},&\text{anti-green}, &h, & \mathbf{p}
# \end{bmatrix}
# \end{equation}
#
# Where $m$ is the on-shell mass. Examples of particles.
#
# (NEED TO CORRECT MY COLOUR EXAMPLES)
#
# Lepton:
#
# \begin{align}
#   e^-_\uparrow &= \begin{bmatrix}
#    m_e, & \frac{1}{2}, & -\frac{1}{2}, & -1,& 0,& -2, & 0, & +1, & \mathbf{p}
#   \end{bmatrix} \\[1em]
#   e^-_\downarrow &= \begin{bmatrix}
#    m_e, & \frac{1}{2}, & -\frac{1}{2}, & -1,& 0,& -2, & 0, & -1, & \mathbf{p}
#   \end{bmatrix}\\[1em]
#   e^+_\uparrow &= \begin{bmatrix}
#     m_e, & \frac{1}{2}, & 0, &+2, &+\frac{1}{2}, &+1, & 0, & +1, & \mathbf{p}
#   \end{bmatrix} \\[1em]
#   e^+_\downarrow &= \begin{bmatrix}
#     m_e, & \frac{1}{2}, & 0, &+2, &+\frac{1}{2}, &+1, & 0, & -1, & \mathbf{p}
#   \end{bmatrix}
# \end{align}
#
# Photon:
#
# \begin{align}
#   \gamma = \begin{bmatrix}
#     0, &1, &0, &0, &0, &0, &0, &h, &\mathbf{p}
#   \end{bmatrix}
# \end{align}
#
# Colour is the number of colour charges the particle has, e.g. quarks have 1 and gluons have 2
#
# ---
#
#
# ### Edge Index (adjacency list):
# A list of doublets that describe which edges connect to which.
#
# e.g.
# ```
# edge_index = [[1,2],[2,1],[1,3],[3,1]]
# ```
#
# ---
#
# ### **Define constants**
#
# Always using natural units in MeV


# lepton masses
m_e = 0.5110
m_mu = 105.6583755
m_tau = 1776

# quark masses
m_up = 2.2
m_down = 4.7
m_charm = 1275
m_strange = 95
m_top = 172760
m_bottom = 4180

# massive bosons
m_W = 80379
m_Z = 91187
m_H = 125100

alpha_QED = 1 / 137
alpha_S = 1
alpha_W = 1e-6
q_e = np.sqrt(4 * np.pi * alpha_QED)
num_edge_feat = 9

"""---
# **Creating Graph Representation Classes**

FeynmanGraph class has been taken from GitHub

"""


def graph_combine(graph1, graph2):
    graph2[1][0] = [
        x + len(graph1[0]) - 1 for x in graph2[1][0]
    ]  # The -1 is included because the nodes are numbered starting from 0
    graph2[1][1] = [x + len(graph1[0]) - 1 for x in graph2[1][1]]

    nodes = graph1[0] + graph2[0]
    edge_index = [graph1[1][0] + graph2[1][0], graph1[1][1] + graph2[1][1]]
    edge_feat = graph1[2] + graph2[2]

    return [nodes, edge_index, edge_feat]


class FeynmanGraph:
    """
    Represents a directed graph using an adjacency list, with support for dynamic
    graph behavior and optional inclusion of a global node (node 0).

    The graph is structured as a set of tuples representing directed edges. Nodes are
    indexed from 1, with the optional global node being the 0th node.

    Features:
    - Add and manage edges and nodes dynamically.
    - Support for directed and undirected edges.
    - Node and edge feature storage and validation.
    - Graph visualization and conversion to DataFrame.

    TODO:
    - Store the M_fi function.
    - Create a DataFrame of datapoints for different momenta.
    - Add documentation for feature length for nodes and edges
    - When adding a new node or edge, initialize the feature as empty, so it can still be accessed
    - Need to change validations to accomodate the above change
    """

    def __init__(self, edges: Iterable[tuple] = None):
        """
        Initializes the FeynmanGraph with an optional set of edges.

        Parameters:
        - edges (Iterable[tuple], optional): A collection of tuples representing
        the edges of the graph. Each tuple is a pair of integers (source, destination).
        """
        self._node_feat_dict = {}
        self._edge_feat_dict = {}
        # confusingly the adj_list is stored as a set, but it is converted to a list when accessed
        self._adj_list = set()
        self._nodes = set()
        if edges:
            self.add_edges(edges)

    def M_fi(self):
        raise Exception("No M_fi function has been defined")

    def get_num_nodes(self) -> int:
        """
        Returns the number of unique nodes in the graph.

        Returns:
        - int: The number of nodes.
        """
        return len(self._nodes)

    def graph_size(self) -> int:
        """
        Returns:
         - int: The number of edges in the graph. Doubled if undirected has been called
        """
        return len(self.edge_index)

    # SECTION - Edge methods
    @property
    def edge_index(self) -> list:
        """
        Returns the edge indices stored as an adjacency list
        """
        return sorted(self._adj_list)

    @edge_index.setter
    @typechecked
    def edge_index(self, edges: Iterable[tuple[int, int]]):
        """set additional edge index. Maybe should delete first?"""
        # del self.edge_index
        self.add_edges(edges)
        self.validate_edge_index()

    @typechecked
    def add_edges(self, edges: Iterable[tuple[int, int]] | tuple[int, int]) -> None:
        """
        Adds multiple edges to the graph. Each edge is a tuple of two integers.

        Parameters:
        - edges (Iterable[tuple[int, int]] | tuple[int, int]): A collection of edges
        or a single edge to be added to the graph.

        Raises:
        - ValueError: If any edge is not a tuple of two integers, or if node indices are invalid.
        """
        if isinstance(edges, tuple) and all(isinstance(e, int) for e in edges):
            edges = [edges]

        for edge in edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError(f"Edge must be a tuple of two integers, got: {edge}")

            for node in edge:
                if not isinstance(node, int):
                    raise ValueError(f"Node in edge must be an integer, got: {node}")

                if node < 0:
                    raise ValueError(f"Node index cannot be negative, got: {node}")

        self.add_edges(edges)
        self._nodes.update(itertools.chain.from_iterable(edges))

        # Initialize node feature if it doesn't exist already
        for e in edges:
            if e not in self._node_feat_dict:
                self._node_feat_dict[e] = []

        # Initialize edge feature if it doesn't exist already

    @edge_index.deleter
    def edge_index(self):
        del self._adj_list

    def validate_edge_index(self):
        """
        Validates the edges in the adjacency list. Checks for proper edge format
        and that edge indices are within valid bounds.

        Raises:
        - ValueError: If an edge is improperly formatted or out of bounds.
        """
        if not self._nodes:
            raise ValueError("No nodes in the graph to validate edges.")

        max_node_index = max(self._nodes)
        for edge in self._adj_list:
            # Check if each edge is a tuple of two integers
            if not (
                isinstance(edge, tuple)
                and len(edge) == 2
                and all(isinstance(n, int) for n in edge)
            ):
                raise ValueError(f"Invalid edge format: {edge}")
            # Check bounds
            if any(
                node_index > max_node_index or node_index < 0 for node_index in edge
            ):
                raise ValueError(
                    f"Edge {edge} is out of bounds. Max node index is {max_node_index} and can't be negative."
                )

    def make_edges_undirected(self):
        """
        Make the graph undirected.
        """
        # Create a set for reverse edges
        reverse_edges = {
            (e[1], e[0]) for e in self._adj_list if (e[1], e[0]) not in self._adj_list
        }

        # Update the adjacency list with reverse edges
        self.add_edges(reverse_edges)

    def connect_global_node(self):
        """Connect a global node to all other nodes"""
        # Create edges from the global node to all other nodes
        global_edges = {(0, n) for n in self._nodes if n != 0}

        # Add all edges to the adjacency list
        self.add_edges(global_edges)

        # Register global node
        self._nodes.add(0)

    def get_adj_dict(self):
        adj_dict = {}
        for start, end in self.edge_index:
            if start in adj_dict:
                adj_dict[start].add(end)
            else:
                adj_dict[start] = {end}
        return adj_dict

    def get_adj_matrix(self):
        pass

    # !SECTION
    # SECTION - Node feature methods
    @property
    def node_feat(self):
        """
        Returns node features.
        """
        # self.validate_node_feat()
        return [self._node_feat_dict[str(i)] for i in sorted(self._nodes)]

    @node_feat.setter
    def node_feat(self, nodes, feats):
        """set node features"""
        # Check nodes and feats are same size
        for n, f in nodes, feats:
            self.add_node_feat(n, f)

    @node_feat.deleter
    def node_feat(self):
        del self._node_feat_dict

    @typechecked
    def add_node_feat(self, node_idx: str, feature: list):
        self._node_feat_dict[node_idx] = feature

    def validate_node_feat(self):
        if not self._nodes:
            raise ValueError("No nodes to validate features for.")

        for node in self._nodes:
            node_key = str(node)
            if node_key not in self._node_feat_dict:
                raise ValueError(f"Missing feature for node {node}")

            feature = self._node_feat_dict[node_key]
            if len(feature) != 3:
                raise ValueError(f"Feature must be of length 3, but got {feature}")
            # Add more specific checks depending on the expected feature format
            if not isinstance(feature, list):
                raise ValueError(
                    f"Invalid feature type for node {node}: {type(feature)}"
                )

    # !SECTION

    # SECTION - Edge feature methods
    @property
    def edge_feat(self):
        """
        return edge features

        FIXME - Currently uses a list repr of self.edge_index. Is the guaranteed to be the same ordered everytime?? We need to ensure that when we call self.edge_index and self.edge_feat that each edge feature corresponds to the correct edge in the adjacency list.
        """
        # self.validate_edge_feat()
        # return [self._edge_feat_dict[e] for e in self.edge_index]
        return []

    @edge_feat.setter
    def edge_feat(self, edges, feats):
        # TODO - Run checks
        for edge, feat in edges, feats:
            self.add_edge_feat(edge, feat)

    @edge_feat.deleter
    def edge_feat(self):
        del self._edge_feat_dict

    def add_edge_feat(self, edge: tuple(int, int), feat):
        """
        TODO - docstring
        """
        self._edge_feat_dict[str(edge)] = feat

    def validate_edge_feat(self):
        if not self.edge_index:
            raise ValueError("No edges to validate features for.")

        for edge in self.edge_index:
            if edge not in self._edge_feat_dict:
                raise ValueError(f"Missing feature for edge {edge}")

            feature = self._edge_feat_dict[edge]
            if len(feature) != 12:
                raise ValueError(f"Feature must be of length 12, but got {feature}")
            # Validate the type of the edge feature
            if not isinstance(feature, list):
                raise ValueError(
                    f"Invalid feature type for edge {edge}: {type(feature)}"
                )

    # !SECTION
    # SECTION - Display methods
    def build_dfs(self):
        print(self.edge_index, self.node_feat, self.edge_feat)
        return (
            pd.DataFrame(self.edge_index),
            pd.DataFrame(self.node_feat),
            pd.DataFrame(self.edge_feat),
        )

    def display_graph(self):
        if not self._adj_list:
            raise ValueError("Cannot display an empty graph.")

        G = nx.Graph()
        adj_dict = self.get_adj_dict()

        if not adj_dict:
            raise ValueError(
                "Adjacency dictionary is empty, unable to construct the graph."
            )

        # Add nodes
        G.add_nodes_from(adj_dict.keys())

        # Add edges
        for node, neighbors in adj_dict.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=500,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight="bold",
        )

        plt.title("Graph Visualization")
        plt.show()
