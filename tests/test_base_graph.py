import numpy as np
from feynman_gnn.base_feynman_graph import FeynmanGraph
from feynman_gnn.particles import ParticleRegistry


class TestBaseFeynmanGraph:
    def test_graph_creation(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (2, 4), (4, 5), (4, 6)])
        for i in range(1, 7):
            graph.add_node_feats({i: [1, 0, 0]})

    def test_graph_dataframe(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2)])
        graph.mfi_squared = lambda p, x: np.random.random(p.shape)
        print(graph.edges)
        graph.build_df(0, 100, 0, 100, 100)

    def test_add_edges(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        assert len(graph.edges) == 3

    def test_add_node_feat(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        graph.add_node_feats({1: [1, 0, 0]})
        assert graph.node_feat[1] == [1, 0, 0]

    def test_graph_edge_index(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        assert graph.edges == [(1, 2), (2, 3), (3, 4)]

    def test_graph_build_df(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        graph.mfi_squared = lambda p, x: np.random.random(p.shape)
        df = graph.build_df(0, 100, 0, 100, 100)

        # TODO - more assertions
        assert len(df) == 10000

    def test_graph_addition(self):
        graph1 = FeynmanGraph()
        graph1.add_edges([(1, 2), (2, 3), (3, 4)])
        graph2 = FeynmanGraph()
        graph2.add_edges([(1, 2), (2, 3), (3, 4)])
        graph = graph1 + graph2
        assert len(graph.edges) == 6

    def test_validations(self):
        FeynmanGraph().validate_graph()

    def test_vertex_check(self):
        E_Minus = ParticleRegistry.get_particle_class("e_minus")()
        graph = FeynmanGraph()
        graph.add_edges([(1, 2), (2, 3), (3, 4)])
        graph.add_node_feats({1: [0, 1, 0]})
        graph.add_edge_feats({(1, 2): E_Minus.features})

        assert not graph.vertex_check()
