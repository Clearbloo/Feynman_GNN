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

    def test_vertex_check(self):
        graph = S_Channel()
        graph.add_node_feat(
            {
                1: [1, 0, 0],
                2: [1, 0, 0],
                3: [0, 1, 0],
                4: [0, 1, 0],
                5: [0, 0, 1],
                6: [0, 0, 1],
            }
        )
        e_minus = ParticleRegistry.get_particle_class("e_minus")()
        e_plus = ParticleRegistry.get_particle_class("e_plus")()
        mu_minus = ParticleRegistry.get_particle_class("mu_minus")()
        mu_plus = ParticleRegistry.get_particle_class("mu_plus")()
        photon = ParticleRegistry.get_particle_class("photon")()
        edge_feat = {
            (1, 3): e_minus.get_features(),
            (2, 3): e_plus.get_features(),
            (3, 4): photon.get_features(),
            (4, 5): mu_minus.get_features(),
            (4, 6): mu_plus.get_features(),
        }
        graph.add_edge_feat(edge_feat)
        assert graph.vertex_check(debug=True)

    def test_build_tree_diagrams(self):
        E_minus: Type[Particle] = ParticleRegistry.get_particle_class("e_minus")()
        E_plus: Type[Particle] = ParticleRegistry.get_particle_class("e_plus")()
        diagrams: List[DataFrame] = build_tree_diagrams(
            E_minus.get_features(),
            E_plus.get_features(),
            E_minus.get_features(),
            E_plus.get_features(),
            T_Channel,
            global_connect=True,
        )
        assert len(diagrams) == 1
        assert diagrams[0].get_num_nodes() == 6
        assert diagrams[0].graph_size() == 5
        assert diagrams[0].node_feat
        assert diagrams[0].edge_index
        assert diagrams[0].edge_feat


if __name__ == "__main__":
    pytest.main()
