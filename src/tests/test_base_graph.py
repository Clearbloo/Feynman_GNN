import sys
import os.path as osp
import numpy as np

# Add the directory to the Python path
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(script_dir))

from base_feynman_graph import FeynmanGraph  # noqa: E402


class TestFeynmanGraph:
    def test_graph_creation(self):
        graph = FeynmanGraph()
        graph.edge_index = [(1, 2), (2, 3), (2, 4), (4, 5), (4, 6)]
        graph.make_edges_undirected()
        for i in range(1, 7):
            graph.add_node_feat({i: [1, 0, 0]})

    def test_graph_dataframe(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2)])
        graph.Mfi_squared = lambda p, x: np.random.random(p.shape)
        print(graph.edge_index)
        graph.build_df(0, 100, 0, 100, 100)

    def test_add_edges(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        assert len(graph.edge_index) == 3

    def test_make_edges_undirected(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        graph.make_edges_undirected()
        assert (2, 1) in graph.edge_index and (3, 2) in graph.edge_index and (4, 3) in graph.edge_index

    def test_add_node_feat(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        graph.add_node_feat({1: [1, 0, 0]})
        assert graph._node_feat_dict[1] == [1, 0, 0]

    def test_graph_edge_index(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        assert graph.edge_index == [(1, 2), (2, 3), (3, 4)]

    def test_graph_node_features(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        graph.add_node_feat({1: [1, 0, 0]})
        assert graph._node_feat_dict[1] == [1, 0, 0]

    def test_graph_build_df(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        graph.Mfi_squared = lambda p, x: np.random.random(p.shape)
        graph.build_df(0, 100, 0, 100, 100)
        # Add assertions for the expected behavior of the build_df method

    def test_graph_addition(self):
        graph1 = FeynmanGraph()
        graph1.add_edges([(1, 2), (2, 3), (3, 4)])
        graph2 = FeynmanGraph()
        graph2.add_edges([(1, 2), (2, 3), (3, 4)])
        graph = graph1 + graph2
        assert len(graph.edge_index) == 6

    def test_validations(self):
        FeynmanGraph().validate_edge_feat()
        FeynmanGraph().validate_node_feat()
        FeynmanGraph().validate_edge_index()