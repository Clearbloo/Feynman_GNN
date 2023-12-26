import pytest
import sys
import os.path as osp
from typing import Type, List
from pandas import DataFrame

# Add the directory to the Python path
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(script_dir))

from feynman_graphs import S_Channel, T_Channel, U_Channel, build_tree_diagrams  # noqa: E402
from particles import ParticleRegistry, Particle  # noqa: E402


class TestFeynmanGraphs:
    def test_s_channel(self):
        s_channel = S_Channel()
        assert s_channel.get_num_nodes() == 6
        s_channel.connect_global_node()
        assert s_channel.get_num_nodes() == 7
        assert s_channel.node_feat
        assert s_channel.edge_index
        assert s_channel.edge_feat

    def test_t_channel(self):
        t_channel = T_Channel()
        assert t_channel.get_num_nodes() == 6
        t_channel.connect_global_node()
        assert t_channel.get_num_nodes() == 7
        assert t_channel.node_feat
        assert t_channel.edge_index
        assert t_channel.edge_feat

    def test_u_channel(self):
        u_channel = U_Channel()
        assert u_channel.get_num_nodes() == 6
        u_channel.connect_global_node()
        assert u_channel.get_num_nodes() == 7
        assert u_channel.node_feat
        assert u_channel.edge_index
        assert u_channel.edge_feat

    def test_build_tree_diagrams(self):
        E_minus: Type[Particle] = ParticleRegistry.get_particle_class("e_minus")
        E_plus: Type[Particle] = ParticleRegistry.get_particle_class("e_plus")
        diagrams: List[DataFrame] = build_tree_diagrams(
            E_minus(), E_plus(), E_minus(), E_plus(), S_Channel, True
        )
        assert len(diagrams) == 1
        assert diagrams[0].get_num_nodes() == 7
        assert diagrams[0].graph_size() == 22
        assert diagrams[0].node_feat
        assert diagrams[0].edge_index
        assert diagrams[0].edge_feat


if __name__ == "__main__":
    pytest.main()
