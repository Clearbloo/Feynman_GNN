import sys
import os.path as osp
import pytest

# Add the directory to the Python path
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(script_dir))

from dataset_builder import build_tree_diagrams  # noqa: E402
from particles import ParticleRegistry, Particle  # noqa: E402
from feynman_graphs import T_Channel  # noqa: E402


class TestDatasetBuilder:
    @pytest.mark.skip(reason="Test is broken. Waiting on vertex_check method to be fixed")
    def test_build_tree_diagrams(self):
        # FIXME - failing test
        E_Plus: type[Particle] = ParticleRegistry.get_particle_class("E_Plus")
        E_Minus: type[Particle] = ParticleRegistry.get_particle_class("E_Minus")

        graphs = build_tree_diagrams(  # noqa: F841
            E_Minus(),
            E_Plus(),
            E_Minus(),
            E_Plus(),
            T_Channel,
            global_connect=True,
        )
        assert len(graphs) == 2
        assert len(graphs[0].edge_index) == 5
        assert len(graphs[1].edge_index) == 5
