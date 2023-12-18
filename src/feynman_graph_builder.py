import numpy as np
from base_feynman_graph import FeynmanGraph

# TODO convert to registry
from particles import (
    Gluon_rbbar,
    Photon,
)

DATASETPATH = "./data"
raw_filepath = f"{DATASETPATH}/raw"

### **Define constants**
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

        # TODO - add edge and node features. Adding in placeholder for now
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


s = S_Channel()
print(s.edge_index, s.edge_feat, s.node_feat)


class T_Channel(FeynmanGraph):
    def __init__(self):
        super().__init__()
        # Add edges
        edges = [(1, 2), (2, 3), (2, 5), (4, 5), (5, 6)]
        self.add_edges(edges)

        # TODO - add edge and node features


class U_Channel(FeynmanGraph):
    def __init__(self):
        super().__init__()
        # Add edges
        edges = [(1, 2), (2, 6), (2, 5), (4, 5), (5, 3)]
        self.add_edges(edges)

        # TODO - add edge and node features


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
    # TODO add this to the base class
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


# def main():

#     # write the matrix element as a function
#     def Mfi_squared(p, theta):
#         return (
#             (q_e**2 * (1 + np.cos(theta))) ** 2 + (q_e**2 * (1 - np.cos(theta))) ** 2
#         ) / 2

#     graph = diagram_builder(
#         E_minus(), E_plus(), Mu_minus(), Mu_plus(), s_channel(), True
#     )
#     df_e_annih = dataframe_builder(
#         0,
#         ang_res=100,
#         p_res=100,
#         p_min=10**3,
#         p_max=10**6,
#         Mfi_squared=Mfi_squared,
#         graph=graph,
#     )

#     # save the file
#     df_e_annih_mu = df_e_annih
#     os.makedirs(raw_filepath, exist_ok=True)
#     df_e_annih_mu.to_csv(path_or_buf=f"{raw_filepath}/QED_data_e_annih_mu.csv")

#     df_e_annih_mu.plot("theta", "y", kind="scatter")

#     # ## **Extending the dataset**
#     #
#     # Repeating for
#     # $$e^-e^+\to e^-e^+$$
#     #
#     # This requires the inclusion of the t-channel diagrams

#     def Mfi_squared(p, theta):
#         """
#         s = p_1^2 + p_2^2 + 2p_1p_2
#         t = p_1^2 + p_3^2 - 2p_1p_3
#         """
#         s = 4 * (m_e**2 + p**2)
#         t = 2 * m_e**2 - 2 * p**2 * (1 - math.cos(theta))
#         u = 2 * m_e**2 - 2 * p**2 * (1 + math.cos(theta))
#         return (
#             2
#             * q_e**4
#             * ((u**2 + t**2) / s**2 + (u**2 + s**2) / t**2 + 2 * u**2 / (s * t))
#         )  # 8*(q_e**4)*(s**4+t**4+u**4)/(4*s**2*t**2)

#     graph_t = diagram_builder(
#         E_minus(), E_plus(), E_minus(), E_plus(), t_channel(), True
#     )
#     graph_s = diagram_builder(
#         E_minus(), E_plus(), E_minus(), E_plus(), s_channel(), True
#     )
#     graph = graph_combine(graph_s, graph_t)
#     df = dataframe_builder(
#         0.5,
#         ang_res=400,
#         p_res=100,
#         p_min=10**3,
#         p_max=10**6,
#         Mfi_squared=Mfi_squared,
#         graph=graph,
#     )
#     df_e_annih_e = df
#     df_e_annih = pd.concat([df_e_annih, df], ignore_index=True)
#     df_merge = df_e_annih

#     # save the file
#     os.makedirs(raw_filepath, exist_ok=True)
#     df_e_annih_e.to_csv(path_or_buf=raw_filepath + "/QED_data_e_annih_e.csv")
#     df_e_annih.to_csv(path_or_buf=raw_filepath + "/QED_data_e_annih.csv")

#     """Other combinations"""

#     def Mfi_squared(p, theta):
#         return (
#             (q_e**2 * (1 + np.cos(theta))) ** 2 + (q_e**2 * (1 - np.cos(theta))) ** 2
#         ) / 2

#     graph = diagram_builder(
#         Mu_minus(), Mu_plus(), E_minus(), E_plus(), s_channel(), True
#     )
#     df = dataframe_builder(
#         0,
#         ang_res=100,
#         p_res=100,
#         p_min=0,
#         p_max=10**6,
#         Mfi_squared=Mfi_squared,
#         graph=graph,
#     )
#     df_annih = df_e_annih
#     df_annih = pd.concat([df_annih, df], ignore_index=True)

#     graph = diagram_builder(
#         Mu_minus(), Mu_plus(), Mu_minus(), Mu_plus(), s_channel(), True
#     )
#     df = dataframe_builder(
#         0,
#         ang_res=100,
#         p_res=100,
#         p_min=10**3,
#         p_max=10**6,
#         Mfi_squared=Mfi_squared,
#         graph=graph,
#     )
#     df_annih = pd.concat([df_annih, df], ignore_index=True)

#     # save the file
#     os.makedirs(raw_filepath, exist_ok=True)
#     df_annih.to_csv(path_or_buf=raw_filepath + "/QED_data_annih.csv")

#     df_annih.plot("theta", "y_norm", kind="scatter")

#     # ## QCD

#     def Mfi_squared(p, theta):
#         """
#         s = p_1^2 + p_2^2 + 2p_1p_2
#         t = p_1^2 + p_3^2 - 2p_1p_3
#         """
#         if p <= m_top:
#             return 0
#         s = 4 * p**2
#         t = m_top**2 - 2 * p**2 + 2 * p * math.cos(theta) * math.sqrt(p**2 - m_top**2)
#         u = m_top**2 - 2 * p**2 - 2 * p * math.cos(theta) * math.sqrt(p**2 - m_top**2)
#         return (
#             16 * alpha_S**4 * (t**4 + u**4 + 2 * m_top**2 * (2 * s - m_top**2)) / (s**2)
#         )

#     graph = diagram_builder_gluon(
#         Up_r(), AntiUp_b(), Top_r(), AntiTop_b(), s_channel(), True
#     )
#     df = dataframe_builder(
#         0,
#         ang_res=1000,
#         p_res=10,
#         p_min=m_top * 0.9,
#         p_max=m_top * 2,
#         Mfi_squared=Mfi_squared,
#         graph=graph,
#     )
#     df_QCD = df

#     # save the file
#     os.makedirs(raw_filepath, exist_ok=True)
#     df_QCD.to_csv(path_or_buf=raw_filepath + "/QCD_data.csv")

#     df_all = pd.concat([df_merge, df_QCD], ignore_index=True)
#     df_all.to_csv(path_or_buf=raw_filepath + "/all_data.csv")

#     df_all.plot("theta", "y_norm", kind="scatter")

#     """#**Testing Sampling**"""

#     w = df_e_annih_mu["y"].astype(float).round(4)
#     w_dict = 1 / w.value_counts()

#     def weight_lookup(x):
#         x = np.round(x, 4)
#         return w_dict[x]

#     df_e_annih_mu["sample_weight"] = df_e_annih_mu["y"].apply(
#         lambda x: weight_lookup(x)
#     )

#     rep_df = df_e_annih_mu.sample(n=1000, weights=df_e_annih_mu["sample_weight"])
#     # print(rep_df['y'])
#     rep_df["y"].hist(bins=10)


if __name__ == "__main__":

    def place_holder():
        return 0

    place_holder()
    # main()
