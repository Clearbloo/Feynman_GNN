import sys
import os.path as osp

# Add the directory to the Python path
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(script_dir))

from feynman_graphs import S_Channel  # noqa: E402


class TestFeynmanGraphs:
    def test_registry(self):
        s_channel = S_Channel()
        assert s_channel.get_num_nodes() == 6
        s_channel.connect_global_node()
        assert s_channel.get_num_nodes() == 7
        assert s_channel.node_feat
        assert s_channel.edge_index
        assert s_channel.edge_feat
