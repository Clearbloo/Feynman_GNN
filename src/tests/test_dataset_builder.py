import sys
import os.path as osp

# Add the directory to the Python path
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(script_dir))

from dataset_builder import build_tree_diagrams  # noqa: E402
from particles import ParticleRegistry, Particle  # noqa: E402
from feynman_graphs import T_Channel  # noqa: E402


class TestDatasetBuilder:
    def test_build_tree_diagrams(self):
        E_Plus: type[Particle] = ParticleRegistry.get_particle_class("E_Plus")()
        E_Minus: type[Particle] = ParticleRegistry.get_particle_class("E_Minus")()

        graphs = build_tree_diagrams(  # noqa: F841
            E_Minus.get_features(),
            E_Plus.get_features(),
            E_Minus.get_features(),
            E_Plus.get_features(),
            T_Channel,
            global_connect=True,
        )
        assert len(graphs) == 1
        assert len(graphs[0].edge_index) == 5
        assert len(graphs[0].node_feat) == 6