import sys
import os.path as osp

# Add the directory to the Python path
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(script_dir))

from feynman_graphs import S_Channel, build_tree_diagrams  # noqa: E402
from particles import ParticleRegistry  # noqa: E402

class TestFeynmanGraphs:
    def test_s_channel(self):
        s_channel = S_Channel()
        assert s_channel.get_num_nodes() == 6
        s_channel.connect_global_node()
        assert s_channel.get_num_nodes() == 7
        assert s_channel.node_feat
        assert s_channel.edge_index
        assert s_channel.edge_feat

    def test_build_tree_diagrams(self):
        E_minus = ParticleRegistry.get_particle_class("e_minus")
        E_plus = ParticleRegistry.get_particle_class("e_plus")
        assert build_tree_diagrams(
            E_minus(), E_plus(), E_minus(), E_plus(), S_Channel, True
        ) == []
