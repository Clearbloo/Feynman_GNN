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

        # graph.connect_global_node()
        graph.make_edges_undirected()

        for i in range(1, 7):
            graph.add_node_feat(i, [1, 0, 0])

    def test_graph_dataframe(self):
        graph = FeynmanGraph()
        graph.add_edges([(1, 2)])
        graph.Mfi_squared = lambda p, x: np.random.random(p.shape)
        print(graph.edge_index)
        graph.build_df(0, 100, 0, 100, 100)
        # TODO - fix
        # graph.create_graph_display()
        # graph.display_graph()
        # graph.close_display()
