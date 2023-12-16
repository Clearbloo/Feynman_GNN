# # **Feynman diagram dataset builder**
#
# Original file is located at
#     https://colab.research.google.com/drive/1Zt6BV0pZzdkWU8Pq2LDjvN4p72Wmg15n


## Standard libraries
import itertools
import math
import os
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

## Imports for plotting
from IPython.display import set_matplotlib_formats

# TODO convert to registry
from particles import (
    AntiTop_b,
    AntiUp_b,
    E_minus,
    E_plus,
    Gluon_rbbar,
    Mu_minus,
    Mu_plus,
    Photon,
    Top_r,
    Up_r,
)
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


# ## Diagram structures
# * s-channel
# * t-channel
# * u-channel
#
# Need to think if i want these to be instances of the FeynmanGraph class or inherited from...


def s_channel():
    initial = [1, 0, 0]
    virtual = [0, 1, 0]
    final = [0, 0, 1]

    # Global node (I happen to be lucky in that these are the same size, 3)
    global_node = [alpha_QED, alpha_W, alpha_S]

    # Node features
    node_features = [global_node, initial, initial, virtual, virtual, final, final]
    num_nodes = len(node_features)
    s_channel = FeynmanGraph(num_nodes)

    # Adjacency lists. The order in which the first and last two nodes are added IS IMPORTANT. initial_1=node 1. final_5=node 5 etc
    s_channel.add_edge(1, 3)  # must connect to first initial state
    s_channel.add_edge(2, 3)  # must connect to second initial state
    s_channel.add_edge(3, 4)
    s_channel.add_edge(4, 5)  # must connect to first final state
    s_channel.add_edge(4, 6)  # must connect to second final state

    s_channel.node_features = node_features


def t_channel():
    initial = [1, 0, 0]
    virtual = [0, 1, 0]
    final = [0, 0, 1]

    # Global node
    global_node = [alpha_QED, alpha_W, alpha_S]

    # Node features
    node_features = [global_node, initial, initial, virtual, virtual, final, final]
    num_nodes = len(node_features)

    # Adjacency lists
    t_chan = FeynmanGraph(num_nodes)
    t_chan.add_edge(1, 3)
    t_chan.add_edge(2, 4)
    t_chan.add_edge(3, 4)
    t_chan.add_edge(3, 5)
    t_chan.add_edge(4, 6)

    edge_index = t_chan

    return node_features, edge_index


def u_channel():
    initial = [1, 0, 0]
    virtual = [0, 1, 0]
    final = [0, 0, 1]

    # Global node
    global_node = [alpha_QED, alpha_W, alpha_S]

    # Node features
    node_features = [global_node, initial, initial, virtual, virtual, final, final]
    num_nodes = len(node_features)

    # Adjacency lists
    u_chan = FeynmanGraph(num_nodes)
    u_chan.add_edge(1, 3)
    u_chan.add_edge(2, 4)
    u_chan.add_edge(3, 4)
    u_chan.add_edge(
        3, 6
    )  # in the u-channel, the first initial state connects to the first final state

    edge_index = u_chan

    return node_features, edge_index


# ## Feynman Diagram builder
#
# A function to create a list of all possible diagrams
#
# 1.   Takes in initial and final particle states
# 2.   Iterates over diagram structures (s,t,u for tree level)
# 3.   Iterates over vertices
# 4.   Removes diagrams that don't follow conservation rules
#
# ---
#
# ## Create Tree-level QED Dataset
#
# We start with simple electron positron to muon antimuon QED tree level scattering. This removes the need for the t-channel diagram. And we only need to consider the s-channel diagram.
#
# The structure is to create a list for each of the features.
#
# Then create a list of lists to represent the data for a singular graph
#
# Then create a stacked list of lists of lists to represent the full dataset which then gets passed to pandas.dataframe
#
# As a numpy array, this will be a 2D array (num_graphs, num_feature_type) with the objects as lists.
#
# I first give the 4 non-zero matrix elements and then include two that are zero.


def vertex_check(vertex, edge_feat, edge_index):
    """
    Function that returns a true or false based on whether quantities are conserved at the specified vertex
    """
    # incoming indices since edge_index[1] are the destinations
    inc_indices = [k for k, x in enumerate(edge_index[1]) if x == vertex]
    inc_edges = [0] * len(edge_feat[0])
    for n in inc_indices:
        inc_edges = [sum(value) for value in zip(edge_feat[n], inc_edges)]

    # outgoing indices since edge_index[0] are the sources
    out_indices = [k for k, x in enumerate(edge_index[0]) if x == vertex]
    out_edges = [0] * len(edge_feat[0])
    for n in out_indices:
        out_edges = [sum(value) for value in zip(edge_feat[n], out_edges)]

    current = [sum(value) for value in zip(inc_edges, [-x for x in out_edges])]
    conservation = [
        current[2]
        + 0.5 * current[3],  # Left charge, weak isospin - 1/2 left hypercharge
        current[4] + 0.5 * current[5],  # "" for the right chiral
        current[6] - current[9],  # red colour charge conservation (red - anti-red)
        current[7] - current[10],  # blue colour charge conservation (blue - anti-blue)
        current[8] - current[11],  # green ""
    ]
    return all(float(charge) == 0.0 for charge in conservation)


def diagram_builder(
    initial_0, initial_1, final_4, final_5, channel, global_connect: bool
):
    """
    Function to make return all possbile diagrams with initial and final states given
    Returns a list allowed graphs, which consist of Feyn_vertex, edge_index and edge_feat
    Changes: should allow feynman diagrams with False to be returned but force them to have matrix element 0; exclude certain vertices e.g. connecting electron to muon
    """
    Feyn_vertex, adj_class = channel
    num_edges = adj_class.graph_size()
    edge_index = adj_class.get_list()
    # check to see if process is kinematically allowed by conserving energy, helicity and momentum (need to add)

    # Given edge features
    incoming_0 = initial_0.get_feat()
    incoming_1 = initial_1.get_feat()
    outgoing_4 = final_4.get_feat()
    outgoing_5 = final_5.get_feat()

    # make empty edge feature list for directed graph
    edge_feat = [0] * num_edges

    # assign initial and final edge feats. NEED TO CHANGE THIS TO SEARCH FOR INIT AND FINAL NODES AS THE INDICES
    edge_feat[0] = incoming_0
    edge_feat[1] = incoming_1
    edge_feat[-2] = outgoing_4
    edge_feat[-1] = outgoing_5

    # create a list of allowed edges to insert between virtual nodes
    graphs = []
    propagators = []
    edge_position = []
    for i in range(len(edge_index[0])):  # len(edge_index[0] is the number of edges)
        # look for virtual nodes connected to virtual nodes
        if Feyn_vertex[edge_index[0][i]] == [0, 1, 0] and Feyn_vertex[
            edge_index[1][i]
        ] == [0, 1, 0]:  # 1-hot encoding for virtual nodes
            # cycle through list of bosons (just photons for now)
            edge_feat[i] = Photon().get_feat()
            if vertex_check(edge_index[0][i], edge_feat, edge_index):
                propagators.append(edge_feat[i])
                edge_position.append(i)
            if not propagators:  # checks to see if the list is empty
                return []

    # cycle through edge_position
    """
  look at edge positions, take all the indices in edge positions
  make lists for each 
  """

    # Connect the global node and make the graph undirected
    if global_connect is True:
        adj_class.connect_global_node()

        # add global node edge features
        num_nodes = len(Feyn_vertex)  # including super node
        for i in range(1, num_nodes):
            global_edge_features = [0] * len(edge_feat[0])
            edge_feat.append(global_edge_features)

    adj_class.undirected()
    edge_index = adj_class.get_list()

    # make the features undirected
    edge_feat += edge_feat
    graphs.append([Feyn_vertex, edge_index, edge_feat])

    return graphs[0]


def diagram_builder_gluon(
    initial_0, initial_1, final_4, final_5, channel, global_connect: bool
):
    """
    Function to make return all possbile diagrams with initial and final states given
    Returns a list allowed graphs, which consist of Feyn_vertex, edge_index and edge_feat
    Changes: should allow feynman diagrams with False to be returned but force them to have matrix element 0; exclude certain vertices e.g. connecting electron to muon
    """
    Feyn_vertex, adj_class = channel
    num_edges = adj_class.graph_size()
    edge_index = adj_class.get_list()
    # check to see if process is kinematically allowed by conserving energy, helicity and momentum (need to add)

    # Given edge features
    incoming_0 = initial_0.get_feat()
    incoming_1 = initial_1.get_feat()
    outgoing_4 = final_4.get_feat()
    outgoing_5 = final_5.get_feat()

    # make empty edge feature list for directed graph
    edge_feat = [0] * num_edges

    # assign initial and final edge feats. NEED TO CHANGE THIS TO SEARCH FOR INIT AND FINAL NODES AS THE INDICES
    edge_feat[0] = incoming_0
    edge_feat[1] = incoming_1
    edge_feat[-2] = outgoing_4
    edge_feat[-1] = outgoing_5

    # create a list of allowed edges to insert between virtual nodes
    graphs = []
    propagators = []
    edge_position = []
    for i in range(len(edge_index[0])):  # len(edge_index[0] is the number of edges)
        # look for virtual nodes connected to virtual nodes
        if Feyn_vertex[edge_index[0][i]] == [0, 1, 0] and Feyn_vertex[
            edge_index[1][i]
        ] == [0, 1, 0]:  # 1-hot encoding for virtual nodes
            # cycle through list of bosons (just photons for now)
            edge_feat[i] = Gluon_rbbar().get_feat()
            if vertex_check(edge_index[0][i], edge_feat, edge_index):
                propagators.append(edge_feat[i])
                edge_position.append(i)
            if not propagators:  # checks to see if the list is empty
                return []

    # cycle through edge_position
    """
  look at edge positions, take all the indices in edge positions
  make lists for each 
  """

    # Connect the global node and make the graph undirected
    if global_connect is True:
        adj_class.connect_global_node()

        # add global node edge features
        num_nodes = len(Feyn_vertex)  # including super node
        for i in range(1, num_nodes):
            global_edge_features = [0] * len(edge_feat[0])
            edge_feat.append(global_edge_features)

    adj_class.undirected()
    edge_index = adj_class.get_list()

    # make the features undirected
    edge_feat += edge_feat
    graphs.append([Feyn_vertex, edge_index, edge_feat])

    return graphs[0]


def dataframe_builder(
    theta_min,
    ang_res,
    p_res,
    p_min,
    p_max,
    Mfi_squared,
    graph,
):
    """
    Function to build a dataframe
    """
    # Setup: First make the dataframe a long list of arrays. 20,000 data points
    momenta_range = np.linspace(p_min, p_max, p_res)
    dataframe = np.empty(shape=(ang_res * p_res, 6), dtype=object)

    # Index to count the graph number
    graph_count = 0

    for p in momenta_range:
        for theta in np.linspace(theta_min, np.pi, ang_res):
            # Graph-level target
            target = Mfi_squared(p, theta)

            # Create the dataframe as an numpy array first. Need to add a way to handle empty graphs
            dataframe[graph_count, 0] = graph[0]
            dataframe[graph_count, 1] = graph[1]
            dataframe[graph_count, 2] = graph[2]
            dataframe[graph_count, 3] = target
            dataframe[graph_count, 4] = p
            dataframe[graph_count, 5] = theta

            # increment the index
            graph_count += 1

    dataframe = pd.DataFrame(
        dataframe,
        columns=["x", "edge_index", "edge_attr", "y", "p", "theta"],
        index=np.arange(0, dataframe.shape[0], 1),
    )
    dataframe["y_scaler"] = dataframe["y"].max()
    dataframe["p_scaler"] = dataframe["p"].max()
    dataframe["y_norm"] = dataframe["y"] / dataframe["y"].max()
    dataframe["p_norm"] = dataframe["p"] / dataframe["p"].max()
    return dataframe


def main():
    return 0

    # write the matrix element as a function
    def Mfi_squared(p, theta):
        return (
            (q_e**2 * (1 + np.cos(theta))) ** 2 + (q_e**2 * (1 - np.cos(theta))) ** 2
        ) / 2

    graph = diagram_builder(
        E_minus(), E_plus(), Mu_minus(), Mu_plus(), s_channel(), True
    )
    df_e_annih = dataframe_builder(
        0,
        ang_res=100,
        p_res=100,
        p_min=10**3,
        p_max=10**6,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )

    # save the file
    df_e_annih_mu = df_e_annih
    os.makedirs(raw_filepath, exist_ok=True)
    df_e_annih_mu.to_csv(path_or_buf=f"{raw_filepath}/QED_data_e_annih_mu.csv")

    df_e_annih_mu.plot("theta", "y", kind="scatter")

    # ## **Extending the dataset**
    #
    # Repeating for
    # $$e^-e^+\to e^-e^+$$
    #
    # This requires the inclusion of the t-channel diagrams

    def Mfi_squared(p, theta):
        """
        s = p_1^2 + p_2^2 + 2p_1p_2
        t = p_1^2 + p_3^2 - 2p_1p_3
        """
        s = 4 * (m_e**2 + p**2)
        t = 2 * m_e**2 - 2 * p**2 * (1 - math.cos(theta))
        u = 2 * m_e**2 - 2 * p**2 * (1 + math.cos(theta))
        return (
            2
            * q_e**4
            * ((u**2 + t**2) / s**2 + (u**2 + s**2) / t**2 + 2 * u**2 / (s * t))
        )  # 8*(q_e**4)*(s**4+t**4+u**4)/(4*s**2*t**2)

    graph_t = diagram_builder(
        E_minus(), E_plus(), E_minus(), E_plus(), t_channel(), True
    )
    graph_s = diagram_builder(
        E_minus(), E_plus(), E_minus(), E_plus(), s_channel(), True
    )
    graph = graph_combine(graph_s, graph_t)
    df = dataframe_builder(
        0.5,
        ang_res=400,
        p_res=100,
        p_min=10**3,
        p_max=10**6,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )
    df_e_annih_e = df
    df_e_annih = pd.concat([df_e_annih, df], ignore_index=True)
    df_merge = df_e_annih

    # save the file
    os.makedirs(raw_filepath, exist_ok=True)
    df_e_annih_e.to_csv(path_or_buf=raw_filepath + "/QED_data_e_annih_e.csv")
    df_e_annih.to_csv(path_or_buf=raw_filepath + "/QED_data_e_annih.csv")

    """Other combinations"""

    def Mfi_squared(p, theta):
        return (
            (q_e**2 * (1 + np.cos(theta))) ** 2 + (q_e**2 * (1 - np.cos(theta))) ** 2
        ) / 2

    graph = diagram_builder(
        Mu_minus(), Mu_plus(), E_minus(), E_plus(), s_channel(), True
    )
    df = dataframe_builder(
        0,
        ang_res=100,
        p_res=100,
        p_min=0,
        p_max=10**6,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )
    df_annih = df_e_annih
    df_annih = pd.concat([df_annih, df], ignore_index=True)

    graph = diagram_builder(
        Mu_minus(), Mu_plus(), Mu_minus(), Mu_plus(), s_channel(), True
    )
    df = dataframe_builder(
        0,
        ang_res=100,
        p_res=100,
        p_min=10**3,
        p_max=10**6,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )
    df_annih = pd.concat([df_annih, df], ignore_index=True)

    # save the file
    os.makedirs(raw_filepath, exist_ok=True)
    df_annih.to_csv(path_or_buf=raw_filepath + "/QED_data_annih.csv")

    df_annih.plot("theta", "y_norm", kind="scatter")

    # ## QCD

    def Mfi_squared(p, theta):
        """
        s = p_1^2 + p_2^2 + 2p_1p_2
        t = p_1^2 + p_3^2 - 2p_1p_3
        """
        if p <= m_top:
            return 0
        s = 4 * p**2
        t = m_top**2 - 2 * p**2 + 2 * p * math.cos(theta) * math.sqrt(p**2 - m_top**2)
        u = m_top**2 - 2 * p**2 - 2 * p * math.cos(theta) * math.sqrt(p**2 - m_top**2)
        return (
            16 * alpha_S**4 * (t**4 + u**4 + 2 * m_top**2 * (2 * s - m_top**2)) / (s**2)
        )

    graph = diagram_builder_gluon(
        Up_r(), AntiUp_b(), Top_r(), AntiTop_b(), s_channel(), True
    )
    df = dataframe_builder(
        0,
        ang_res=1000,
        p_res=10,
        p_min=m_top * 0.9,
        p_max=m_top * 2,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )
    df_QCD = df

    # save the file
    os.makedirs(raw_filepath, exist_ok=True)
    df_QCD.to_csv(path_or_buf=raw_filepath + "/QCD_data.csv")

    df_all = pd.concat([df_merge, df_QCD], ignore_index=True)
    df_all.to_csv(path_or_buf=raw_filepath + "/all_data.csv")

    df_all.plot("theta", "y_norm", kind="scatter")

    """#**Testing Sampling**"""

    w = df_e_annih_mu["y"].astype(float).round(4)
    w_dict = 1 / w.value_counts()

    def weight_lookup(x):
        x = np.round(x, 4)
        return w_dict[x]

    df_e_annih_mu["sample_weight"] = df_e_annih_mu["y"].apply(
        lambda x: weight_lookup(x)
    )

    rep_df = df_e_annih_mu.sample(n=1000, weights=df_e_annih_mu["sample_weight"])
    # print(rep_df['y'])
    rep_df["y"].hist(bins=10)


if __name__ == "__main__":
    main()