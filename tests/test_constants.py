import sys
import os.path as osp

# Add the directory to the Python path
script_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(script_dir))

import constants  # noqa: E402


class TestConstants:
    def test_lepton_masses(self):
        assert constants.m_e == 0.5110
        assert constants.m_mu == 105.6583755
        assert constants.m_tau == 1776
