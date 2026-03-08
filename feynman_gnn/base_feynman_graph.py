from __future__ import annotations

import itertools
import warnings
from typing import Annotated, Iterable, Literal, Callable

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from feynman_gnn.error_classes import GraphConstructionError, InvalidEdgeError
from matplotlib_inline.backend_inline import set_matplotlib_formats
from pydantic import AfterValidator, BaseModel, validate_call

set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0


@validate_call
def is_contiguous_range(s: set[int]):
    if s == set():
        # Must have at least 1 element
        return False

    return s == set(range(max(s) + 1))


def not_implemented():
    raise NotImplementedError("No M_fi function has been defined")


class FeynmanGraph(BaseModel):
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

    _node_feat_dict: dict[int, list[Literal[0] | Literal[1]]] = {0: [0, 0, 0]}
    _edge_feat_dict: dict[tuple[int, int], list[Literal[0] | Literal[1]]] = {}
    # Confusingly the adj_list is stored as a set
    _adj_list: set = set()
    _nodes: Annotated[set, AfterValidator(is_contiguous_range)] = set([0])
    _mfi_squared: Callable = lambda *args, **kwargs: not_implemented()

    def __repr__(self):
        return f"FeynmanGraph(num_nodes={self.num_nodes},node_feat={self.node_feat}, edge_feat={self.edge_feat}"

    def __add__(self, other: FeynmanGraph):
        # Check if the other object is also an instance of FeynmanGraph
        if not isinstance(other, FeynmanGraph):
            raise TypeError(
                "Unsupported operand type(s) for +: 'FeynmanGraph' and '{}'".format(
                    type(other).__name__
                )
            )

        result = FeynmanGraph()

        # Relabel the nodes of the other graph
        new_nodes = set([node + self.num_nodes for node in other._nodes])
        result.add_nodes(self.nodes)
        result.add_nodes(new_nodes)

        new_edges = [
            (edge[0] + self.num_nodes, edge[1] + self.num_nodes) for edge in other.edges
        ]
        result.add_edges(self.edges)
        result.add_edges(new_edges)

        new_node_features = {node: other._node_feat_dict[node] for node in other._nodes}
        result.add_node_feats(self.node_feat)
        result.add_node_feats(new_node_features)

        new_edge_features = {
            (
                edge[0] + self.num_nodes,
                edge[1] + self.num_nodes,
            ): other._edge_feat_dict[edge]
            for edge in other._edge_feat_dict
        }
        result.add_edge_feats(self.edge_feat)
        result.add_edge_feats(new_edge_features)

        return result

    @property
    def mfi_squared(self) -> Callable:
        return self._mfi_squared

    @mfi_squared.setter
    def mfi_squared(self, func):
        self._mfi_squared = func

    @property
    def num_nodes(self) -> int:
        """
        Returns the number of unique nodes in the graph.

        Returns:
        - int: The number of nodes.
        """
        return len(self._nodes)

    @property
    def graph_size(self) -> int:
        """
        Returns:
         - int: The number of edges in the graph. Doubled if undirected has been called
        """
        return len(self._adj_list)

    def validate_graph(self):
        """
        Entry point for validation. Validates features, edges and nodes
        """
        self._validate_edge_feat()
        self._validate_node_feats()
        self._validate_edges(self.edges)

    def vertex_check(self, debug=False) -> bool:
        """
        Checks if all vertices conserves quantities

        Checked quantities:
        - Electric charge
        - Colour charge

        TODO - also need to check:
        - Momentum
        - Weak isospin
        - Hypercharge

        This function is the reason that the graphs must be directed, so that we can tell which edges are incoming and outgoing.
        """
        edge_feat_dict = self._edge_feat_dict
        edge_index = self.edges

        for vertex in self._nodes:
            if self._node_feat_dict[vertex] != [0, 1, 0]:
                continue
            inc_edges = [e for e in edge_index if e[1] == vertex]
            out_edges = [e for e in edge_index if e[0] == vertex]

            inc_current = [edge_feat_dict[e] for e in inc_edges]
            inc_current = np.sum(inc_current, axis=0)
            out_current = [edge_feat_dict[e] for e in out_edges]
            out_current = np.sum(out_current, axis=0)

            # inc_currents - out_currents
            current = np.subtract(inc_current, out_current)

            # Check conservations
            # Left charge, weak isospin - 1/2 left hypercharge
            # "" for the right chiral particles
            # Red colour charge conservation
            # Blue
            # Green
            conservation = [
                current[2] + 0.5 * current[3],
                current[4] + 0.5 * current[5],
                current[6] - current[9],
                current[7] - current[10],
                current[8] - current[11],
            ]
            if debug:
                print(f"Vertex {vertex} conserves {conservation}")

            if not all(float(charge) == 0.0 for charge in conservation):
                return False
        return True

    # SECTION - Node methods
    @property
    def nodes(self) -> list:
        return sorted(self._nodes)

    @nodes.setter
    def nodes(self, nodes):
        try:
            iter(nodes)
        except TypeError:
            raise TypeError("Nodes must be iterable")

        if not all(isinstance(n, int) for n in nodes):
            raise TypeError("Nodes must be integers")

        self._nodes = nodes

    @nodes.deleter
    def nodes(self):
        self._nodes = set()

    def add_nodes(self, nodes: Iterable[int]):
        self._nodes.update(nodes)

    # !SECTION

    # SECTION - Edge methods
    @property
    def edges(self) -> list:
        """
        Returns the edge indices stored as an adjacency list
        """
        return sorted(self._adj_list)

    @edges.setter
    def edges(self, edges):
        self._validate_edges(edges)
        self._adj_list = edges

    @edges.deleter
    def edges(self):
        self._adj_list = set()

    def add_edges(self, edges: Iterable[tuple[int, int]]):
        """
        Adds new edges to edge index with a new set of edges.

        Parameters:
            edges (Iterable[tuple[int, int]] | dict[int: int]): The new set of edges to add the existing edge index. Can be given as a list of tuples with first argument as the source node and second as the destination node or as a dict of node indices where the key is the source node and the values are the destinations.
        """
        self._nodes.update(itertools.chain.from_iterable(edges))

        self._validate_edges(edges)
        self._adj_list.update(edges)
        self._validate_edges(self.edges)

        # Initialize node feature if it doesn't exist already
        self.add_node_feats(
            {node: [] for node in self._nodes if node not in self._node_feat_dict}
        )
        self._validate_node_feats()

        # Initialize edge feature if it doesn't exist already
        self.add_edge_feats(
            {edge: [] for edge in edges if edge not in self._edge_feat_dict}
        )
        self._validate_edge_feat()

    def _validate_edges(self, edges):
        """
        Validates the edges. Checks for proper edge format and that edge indices are within valid bounds.

        Raises:
        - ValueError: If an edge is improperly formatted or out of bounds.
        """

        for edge in edges:
            # Check if each edge is a tuple of two integers
            if not (
                isinstance(edge, tuple)
                and len(edge) == 2
                and all(isinstance(n, int) for n in edge)
            ):
                raise InvalidEdgeError(
                    f"Edge must be a tuple of two integers, got: {edge}"
                )
            # Check bounds
            if any(
                node_index > self.num_nodes or node_index < 0 for node_index in edge
            ):
                raise InvalidEdgeError(
                    f"Edge {edge} is out of bounds. Max node index is {self.num_nodes} and can't be negative."
                )

    def connect_global_node(self):
        """Connect a global node to all other nodes"""
        self.add_edges({(0, n) for n in self._nodes if n != 0})

    def get_adj_dict(self):
        adj_dict = {}
        for start, end in self.edges:
            if start in adj_dict:
                adj_dict[start].add(end)
            else:
                adj_dict[start] = {end}
        return adj_dict

    def get_adj_matrix(self):
        raise NotImplementedError()

    # !SECTION

    # SECTION - Node feature methods
    @property
    def node_feat(self):
        """
        Returns node features as a list
        """
        return self._node_feat_dict

    @node_feat.deleter
    def node_feat(self):
        self._node_feat_dict = {}

    def add_node_feats(self, feats: dict[int, list]):
        self._node_feat_dict.update(feats)
        self._validate_node_feats()

    def _validate_node_feats(self):
        """
        Function to test a given set of node features for validity. If no node features are given, the node features of the graph are tested by default.
        """
        feats = self._node_feat_dict
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
        return self._edge_feat_dict

    @edge_feat.deleter
    def edge_feat(self):
        self._edge_feat_dict = {}

    def add_edge_feats(self, edge_feats: dict[tuple[int, int], list]):
        """
        TODO - docstring
        """
        self._edge_feat_dict.update(edge_feats)
        self._validate_edge_feat()

    def _validate_edge_feat(self):
        """
        Function to test a given set of edge features for validity. If no edge features are given, the edge features of the graph are tested by default.
        """
        feats = self._edge_feat_dict

        if not self.edges:
            warnings.warn("No edges to validate features for.")

        for edge in self.edges:
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
        target_grid = self.mfi_squared(p_grid, theta_grid)

        # Flatten the grids
        flat_p = p_grid.flatten()
        flat_theta = theta_grid.flatten()
        flat_target = target_grid.flatten()

        # Prepare data for DataFrame
        data = {
            "x": [self.node_feat] * len(flat_p),
            "edge_index": [self.edges] * len(flat_p),
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
        if self.figure and self.figure.canvas.manager:
            self.figure.canvas.manager.window.update()
            self.figure.show()

    def close_display(self):
        plt.close()
