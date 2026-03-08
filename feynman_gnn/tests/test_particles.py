import sys
import os.path as osp

# Add the directory to the Python path
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(script_dir))

from particles import ParticleRegistry  # noqa: E402


class TestParticles:
    def test_registry(self):
        ParticleRegistry.get_particle_class("e_minus")
        assert ParticleRegistry.list_particles() == [
            "e_minus",
            "e_plus",
            "mu_minus",
            "mu_plus",
            "up_r",
            "antiup_b",
            "down_r",
            "charm_r",
            "strange_r",
            "top_r",
            "antitop_b",
            "bottom_r",
            "photon",
            "gluon_rbbar",
            "w_plus",
            "w_minus",
            "z_0",
            "higgs",
        ], "Particle missing/added!! (New physics??)"
