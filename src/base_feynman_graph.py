## Standard libraries
import itertools
import numpy as np
import warnings
from typing import Iterable, Dict
from error_classes import InvalidEdgeError

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from error_classes import GraphConstructionError

## Imports for plotting
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0

class FeynmanGraph:
    """
    Represents a directed graph using an adjacency list, with support for dynamic
    graph behavior and optional inclusion of a global node (node 0).

    The graph is structured as a set of tuples representing directed edges. Nodes are
    indexed from 1, with the optional global node being the 0th node.

    Main properties:
    - edge_index: The edge index stored as an adjacency list.
    - node_feat: The node features stored as a list.
    - edge_feat: The edge features stored as a list.

    Additional helper methods:
    - add_edges(): Add edges to the graph.
    - add_node_feat(): Add node features to the graph.
    - add_edge_feat(): Add edge features to the graph.

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

    def __init__(self):
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
        # TODO - Make some init values that pass the validations
        # self.validate_graph()

    def Mfi_squared(self, p, theta):
        raise Exception("No M_fi function has been defined")

    def get_num_nodes(self) -> int:
        """
        Returns the number of unique nodes in the graph.

        TODO - refactor this method to be a variable/propery instead of a method that is continuously updated when new nodes are added to the graph.

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
    
    def validate_graph(self):
        """
        TODO - docstring
        """
        self.validate_edge_feat()
        self.validate_node_feat()
        self.validate_edge_index()

    # SECTION - Edge methods
    @property
    def edge_index(self) -> list:
        """
        Returns the edge indices stored as an adjacency list
        """
        return sorted(self._adj_list)

    @edge_index.setter
    def edge_index(self, edges: Iterable[tuple[int, int]] | Dict[int, int] | tuple[int, int]):
        """
        Adds new edges to edge index with a new set of edges.

        Parameters:
            edges (Iterable[tuple[int, int]] | dict[int: int]): The new set of edges to add the existing edge index. Can be given as a list of tuples with first argument as the source node and second as the destination node or as a dict of node indices where the key is the source node and the values are the destinations.

        Behaviours:
            - If edges is a dict, it is converted to a list of tuples.
            - Assigns the new edges to the graph.
            - Validates the new edges.

        Returns:
            None
        """
        if isinstance(edges, dict):
            edges = [(k, v) for k, v in edges.items()]

        self.add_edges(edges)


    @edge_index.deleter
    def edge_index(self):
        self._adj_list = set()
        self._nodes = set()

    def add_edges(self, edges: Iterable[tuple[int, int]] | dict[int, list[int]] | tuple[int, int]):
        """
        Adds multiple edges to the graph. Each edge is a tuple of two integers.

        Parameters:
        - edges (Iterable[tuple[int, int]] | tuple[int, int]) | dict[int: list[int]]: A collection of edges as an iterable or a dict. If given as an iterable, each edge is a tuple of two integers. If given as a dict, the keys are the source nodes and the values are the destination nodes. Can also be a single edge given as a tuple.

        Raises:
        - ValueError: If any edge is not a tuple of two integers, or if node indices are invalid.
        """
        if isinstance(edges, tuple) and all(isinstance(e, int) for e in edges):
            edges = [edges]
        elif isinstance(edges, dict):
            edges = [(k, v) for k, v in edges.items()]
        self._adj_list.update(edges)
        self._nodes.update(itertools.chain.from_iterable(edges))
        self.validate_edge_index()

        # Initialize node feature if it doesn't exist already      
        self.add_node_feat({node: [] for node in self._nodes if node not in self._node_feat_dict})
        self.validate_node_feat()

        # Initialize edge feature if it doesn't exist already
        self.add_edge_feat({edge: [] for edge in edges if edge not in self._edge_feat_dict})
        self.validate_edge_feat()

    def validate_edge_index(self):
        """
        Validates the edges. Checks for proper edge format and that edge indices are within valid bounds.

        Raises:
        - ValueError: If an edge is improperly formatted or out of bounds.
        """
        edges = self._adj_list

        if not self._nodes:
            warnings.warn("No nodes in the graph to validate edges.")

        max_node_index = max(self._nodes or [0])
        for edge in edges:
            # Check if each edge is a tuple of two integers
            if not (
                isinstance(edge, tuple)
                and len(edge) == 2
                and all(isinstance(n, int) for n in edge)
            ):
                raise InvalidEdgeError(f"Edge must be a tuple of two integers, got: {edge}")
            # Check bounds
            if any(
                node_index > max_node_index or node_index < 0 for node_index in edge
            ):
                raise InvalidEdgeError(
                    f"Edge {edge} is out of bounds. Max node index is {max_node_index} and can't be negative."
                )
            # Check the nodes given in the tuple elements
            for node in edge:
                if not isinstance(node, int):
                    raise ValueError(f"Node in edge must be an integer, got: {node}")

                if node < 0:
                    raise ValueError(f"Node index cannot be negative, got: {node}")

    def make_edges_undirected(self):
        """
        Make the graph undirected by adding reverse edges to the adjacency list and updating the edge features.
        """
        # Create a set for reverse edges
        reverse_edges = {
            (e[1], e[0]) for e in self._adj_list if (e[1], e[0]) not in self._adj_list
        }

        # Update the adjacency list with reverse edges
        self.add_edges(reverse_edges)

        # Update the edge features
        self.add_edge_feat({e: self._edge_feat_dict[e] for e in reverse_edges})

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
        Returns node features as a list
        """
        return [self._node_feat_dict[i] for i in sorted(self._nodes)]

    @node_feat.setter
    def node_feat(self, feats: dict[int: list] | list[list]):
        """
        Setter method for the node features of the Feynman graph. Adds new node features to the existing node features.

        Args:
            feats (dict[int: list] | list[list]): A dictionary mapping node indices to feature lists or a list of feature lists, assumed to be in the correct order.
        
        # TODO add docs to explain what the correct order is

        Raises:
            ValueError: If a node index is not initialized (edges not created).

        """
        if isinstance(feats, list):
            feats = {i: feat for i, feat in sorted(self._nodes)}
        
        for node in feats:
            if int(node) not in self._nodes:
                raise ValueError(f"Node {node} not initialized. Need to make the edges first.")

        self.add_node_feat(feats)

    @node_feat.deleter
    def node_feat(self):
        self._node_feat_dict = {}

    def add_node_feat(self, feats: dict[int: list]):
        self._node_feat_dict.update(feats)
        self.validate_node_feat()

    def validate_node_feat(self):
        """
        Function to test a given set of node features for validity. If no node features are given, the node features of the graph are tested by default.
        """
        feats=self._node_feat_dict
        if not self._nodes:
            warnings.warn("No nodes to validate features for.")

        for node in self._nodes:
            if node not in feats:
                raise ValueError(f"Missing feature for node {node}")

            feature = feats[node]
            if len(feature) != 3:
                if feature == []:
                    pass
                else:
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
        Current fix is by calling sorted() on sets
        Alternatively could use a dict
        """
        return [self._edge_feat_dict[e] for e in self.edge_index]

    @edge_feat.setter
    def edge_feat(self, feats: dict):
        self.add_edge_feat(feats)

    @edge_feat.deleter
    def edge_feat(self):
        self._edge_feat_dict = {}

    def add_edge_feat(self, edge_feats: dict[tuple[int,int]: list]):
        """
        TODO - docstring
        """
        self._edge_feat_dict.update(edge_feats)
        self.validate_edge_feat()

    def validate_edge_feat(self):
        """
        Function to test a given set of edge features for validity. If no edge features are given, the edge features of the graph are tested by default.
        """
        feats=self._edge_feat_dict

        if not self.edge_index:
            warnings.warn("No edges to validate features for.")

        for edge in self.edge_index:
            if edge not in feats:
                raise ValueError(f"Missing feature for edge {edge}")

            feature = feats[edge]
            if len(feature) != 12:
                if feature == []:
                    pass
                else:
                    raise ValueError(f"Feature must be of length 12, but got {feature}")
            # Validate the type of the edge feature
            if not isinstance(feature, list):
                raise ValueError(
                    f"Invalid feature type for edge {edge}: {type(feature)}"
                )

    # !SECTION

    def build_df(
        self, theta_min=0, ang_res=100, p_min=0, p_max=1e9, p_res=100
    ) -> pd.DataFrame:
        """
        Function to build a dataframe
        """
        # Vectorized setup
        p_values = np.linspace(p_min, p_max, p_res)
        theta_values = np.linspace(theta_min, np.pi, ang_res)
        p_grid, theta_grid = np.meshgrid(p_values, theta_values, indexing="ij")
        target_grid = self.Mfi_squared(p_grid, theta_grid)

        # Flatten the grids
        flat_p = p_grid.flatten()
        flat_theta = theta_grid.flatten()
        flat_target = target_grid.flatten()

        # Prepare data for DataFrame
        data = {
            "x": [self.node_feat] * len(flat_p),
            "edge_index": [self.edge_index] * len(flat_p),
            "edge_attr": [self.edge_feat] * len(flat_p),
            "y": flat_target,
            "p": flat_p,
            "theta": flat_theta,
        }

        # Create DataFrame
        dataframe = pd.DataFrame(data)

        self.dataframe = dataframe
        return dataframe

    def normalize_df(self):
        self.dataframe["y_max"] = self.dataframe["y"].max()
        self.dataframe["p_max"] = self.dataframe["p"].max()
        self.dataframe["y_norm"] = self.dataframe["y"] / self.dataframe["y"].max()
        self.dataframe["p_norm"] = self.dataframe["p"] / self.dataframe["p"].max()
        return self.dataframe
    
    def __repr__(self):
        return f"FeynmanGraph(num_nodes={self.get_num_nodes()},node_feat={self.node_feat}, edge_feat={self.edge_feat}"

    def __add__(self, other: "FeynmanGraph"):
        # Check if the other object is also an instance of FeynmanGraph
        if not isinstance(other, FeynmanGraph):
            raise TypeError("Unsupported operand type(s) for +: 'FeynmanGraph' and '{}'".format(type(other).__name__))
        
        result = FeynmanGraph()

        # Relabel the nodes of the other graph
        new_nodes = [node + self.get_num_nodes() for node in other._nodes]
        del other._nodes
        other._nodes = set(new_nodes)

        new_edges = [(edge[0] + self.get_num_nodes(), edge[1] + self.get_num_nodes()) for edge in other.edge_index]
        del other.edge_index
        other.edge_index = new_edges

        new_node_features = {node: other._node_feat_dict[node] for node in other._nodes}
        del other.node_feat
        other.node_feat = new_node_features

        new_edge_features = {(edge[0] + self.get_num_nodes(), edge[1] + self.get_num_nodes()): other._edge_feat_dict[edge] for edge in other._edge_feat_dict}
        del other.edge_feat
        other.edge_feat = new_edge_features


        # Add the edges
        result.edge_index = self.edge_index + other.edge_index

        # REVIEW - I think it would be better to use self.node_feat and add the lists in these next two lines, but I'm afraid the order of the nodes will be wrong

        # Add the node features
        result.node_feat = {**self._node_feat_dict, **other._node_feat_dict} 

        # Add the edge features
        result.edge_feat = {**self._edge_feat_dict, **other._edge_feat_dict}
        
        return result


class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph
        self.figure = None

    def create_graph_display(self, display: bool = False):
        adj_dict = self.graph.get_adj_dict()

        if not adj_dict:
            raise GraphConstructionError(
                "Adjacency dictionary is empty, unable to construct the graph."
            )

        G = nx.Graph()

        # Add nodes and edges
        G.add_nodes_from(adj_dict.keys())
        G.add_edges_from(
            (node, neighbor)
            for node, neighbors in adj_dict.items()
            for neighbor in neighbors
        )

        # Create a Matplotlib figure and store it as an attribute
        self.figure, ax = plt.subplots()

        # Draw the graph using the 'ax' object
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=500,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight="bold",
        )

        ax.set_title("Graph Visualization")

        if display:
            plt.show()

    def display_graph(self):
        if self.figure:
            self.figure.canvas.manager.window.update()
            self.figure.show()

    def close_display(self):
        plt.close()