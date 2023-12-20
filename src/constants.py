import numpy as np
# **Define constants**
# Always using natural units in MeV


# lepton masses
m_e = 0.5110
m_mu = 105.6583755
m_tau = 1776

# quark masses
m_up = 2.2
m_down = 4.7
m_charm = 1275
m_strange = 95
m_top = 172760
m_bottom = 4180

# massive bosons
m_W = 80379
m_Z = 91187
m_H = 125100

alpha_QED = 1 / 137
alpha_S = 1
alpha_W = 1e-6
q_e = np.sqrt(4 * np.pi * alpha_QED)
num_edge_feat = 9
