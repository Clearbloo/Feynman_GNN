from base_feynman_graph import FeynmanGraph
from typing import List
from pandas import DataFrame
# import constants
from particles import ParticleRegistry

DATASETPATH = "./data"
raw_filepath = f"{DATASETPATH}/raw"


def graph_combine(graph1, graph2):
    graph2[1][0] = [
        x + len(graph1[0]) - 1 for x in graph2[1][0]
    ]  # The -1 is included because the nodes are numbered starting from 0
    graph2[1][1] = [x + len(graph1[0]) - 1 for x in graph2[1][1]]

    nodes = graph1[0] + graph2[0]
    edge_index = [graph1[1][0] + graph2[1][0], graph1[1][1] + graph2[1][1]]
    edge_feat = graph1[2] + graph2[2]

    return [nodes, edge_index, edge_feat]


# ## Diagram structures
# * s-channel
# * t-channel
# * u-channel
#
# Need to think if i want these to be instances of the FeynmanGraph class or inherited from...
# Also need the specific diagrams to be either an instance or inherited from a classs


class S_Channel(FeynmanGraph):
    def __init__(self):
        super().__init__()
        # Add edges
        edges = [(1, 3), (2, 3), (3, 4), (4, 5), (4, 6)]
        self.add_edges(edges)

        initial = [1, 0, 0]
        virtual = [0, 1, 0]
        final = [0, 0, 1]
        # Bad idea to have the global node with a different context to the 1-hot
        # encodings of the individual nodes. Commenting out for now
        # global_node = [alpha_QED, alpha_W, alpha_S]

        self.node_feat = {
            1: initial,
            2: initial,
            3: virtual,
            4: virtual,
            5: final,
            6: final,
        }


class T_Channel(FeynmanGraph):
    def __init__(self):
        super().__init__()
        # Add edges
        edges = [(1, 2), (2, 3), (2, 5), (4, 5), (5, 6)]
        self.add_edges(edges)

        initial = [1, 0, 0]
        virtual = [0, 1, 0]
        final = [0, 0, 1]

        self.node_feat = {
            1: initial,
            2: virtual,
            3: final,
            4: initial,
            5: virtual,
            6: final,
        }

        print(initial, virtual, final)

class U_Channel(FeynmanGraph):
    def __init__(self):
        super().__init__()
        # Add edges
        edges = [(1, 2), (2, 6), (2, 5), (4, 5), (5, 3)]
        self.add_edges(edges)

        initial = [1, 0, 0]
        virtual = [0, 1, 0]
        final = [0, 0, 1]
        # Bad idea to have the global node with a different context to the 1-hot
        # encodings of the individual nodes. Commenting out for now
        # global_node = [alpha_QED, alpha_W, alpha_S]

        self.node_feat = {
            1: initial,
            2: virtual,
            3: final,
            4: initial,
            5: virtual,
            6: final,
        }


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


def build_tree_diagrams(
    initial_1,
    initial_2,
    final_5,
    final_6,
    channel: FeynmanGraph,
    global_connect: bool,
)-> List[DataFrame]:
    """
    Function to make return all possbile diagrams with given initial and final states.

    Behaviour
    ---
    Creates the graph base, by assigning the edge features.
    Cycles through possible propagators and checks if valid
    Returns list of all allowed ones

    Returns
    ---
    Returns a list allowed graphs, which consist of Feyn_vertex, edge_index and edge_feat
    Changes: should allow feynman diagrams with False to be returned but force them to have matrix element 0; exclude certain vertices e.g. connecting electron to muon
    """

    graph: FeynmanGraph = channel()

    # TODO - check to see if process is kinematically allowed by conserving energy, helicity and momentum (need to add)
    edge_feats = {
        1: initial_1,
        2: initial_2,
        5: final_5,
        6: final_6,
    }
    graph.add_edge_feat(edge_feats)

    # create a list of allowed edges to insert between virtual nodes
    graphs = []

    # look for virtual nodes connected to virtual nodes
    for e in graph.edge_index:
        if e[0] == [0,1,0] and e[1] == [0,1,0]:
            graph.edge_feat[e] = ParticleRegistry.get_particle_class("photon")

    # cycle through edge_position
    """
    look at edge positions, take all the indices in edge positions
    make lists for each 
    """

    # Connect the global node and make the graph undirected
    if global_connect is True:
        graph.connect_global_node()

    graph.make_edges_undirected()

    # make the features undirected
    graphs.append(graph)

    return graphs


def diagram_builder_gluon(
    initial_0,
    initial_1,
    final_4,
    final_5,
    channel,
    global_connect: bool,
) -> List[DataFrame]:
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
            edge_feat[i] = ParticleRegistry.get_particle_class("gluon_rbbar").get_feat()
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
