import numpy as np

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


class ParticleRegistry:
    """TODO Might make one of these to access every particle"""

    pass


class Particle:
    """
    edge features vector
    l=[m,S,LIW3,LY,RIW3,RY,colour,h]
    masses in subsequent classes are given in MeV
    """

    def __init__(
        self,
        m: float,
        S: float,
        LIW: float,
        LY: float,
        RIW: float,
        RY: float,
        colour: list = [0, 0, 0],
        anti_colour: list = [0, 0, 0],
    ):
        self.LIW = LIW
        self.LY = LY
        self.RIW = RIW
        self.RY = RY

        self.feat = [
            m,
            S,
            LIW,
            LY,
            RIW,
            RY,
            colour[0],
            colour[1],
            colour[2],
            anti_colour[0],
            anti_colour[1],
            anti_colour[2],
        ]

    def get_feat(self):
        return self.feat

    def anti_particle(self):
        self.feat[2] = -self.RIW
        self.feat[3] = -self.RY
        self.feat[4] = -self.LIW
        self.feat[5] = -self.LY

    def print_feat(self):
        print(self.feat)


## ANCHOR Lepton classes:
# (need to include tau and all neutrinos still)
class E_minus(Particle):
    """
    Class to construct the edge feautres of an electron
    l=[m,S,IW3,Y,colour,h]
    """

    def __init__(self):
        """
        h = helicity
        p = 3-momentum vector
        """
        Particle.__init__(self, m=m_e, S=0.5, LIW=-0.5, LY=-1, RIW=0, RY=-2)


class E_plus(Particle):
    """
    Class to construct the edge feautres of a positron
    """

    def __init__(self):
        """
        h = helicity
        p = 3-momentum vector
        """
        Particle.__init__(self, m=m_e, S=0.5, LIW=0, LY=2, RIW=0.5, RY=1)


class Mu_minus(Particle):
    """
    Class to construct the edge feautres of a muon
    l=[m,S,IW3,Y,colour,h,p]
    """

    def __init__(self):
        """
        h = helicity
        p = 3-momentum vector
        """
        Particle.__init__(self, m=m_mu, S=0.5, LIW=-0.5, LY=-1, RIW=0, RY=-2)


class Mu_plus(Particle):
    """
    Class to construct the edge feautres of an anti-muon
    """

    def __init__(self):
        """
        h = helicity
        p = 3-momentum vector
        """
        Particle.__init__(self, m=m_mu, S=0.5, LIW=0, LY=2, RIW=0.5, RY=1)


## ANCHOR Quark Classes
class Up_r(Particle):
    """
    edge features of an up quark
    """

    def __init__(self):
        Particle.__init__(
            self, m=m_up, S=0.5, LIW=0.5, LY=1 / 3, RIW=0, RY=4 / 3, colour=[1, 0, 0]
        )


class AntiUp_b(Particle):
    """
    edge features of an up quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            m=m_up,
            S=0.5,
            LIW=0,
            LY=-4 / 3,
            RIW=-0.5,
            RY=-1 / 3,
            anti_colour=[0, 1, 0],
        )


class Down_r(Particle):
    """
    edge features of a down quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            m=m_down,
            S=0.5,
            LIW=-0.5,
            LY=1 / 3,
            RIW=0,
            RY=-2 / 3,
            colour=[1, 0, 0],
        )


class Charm_r(Particle):
    """
    edge features of a charm quark
    """

    def __init__(self):
        Particle.__init__(
            self, m=m_charm, S=0.5, LIW=0.5, LY=1 / 3, RIW=0, RY=4 / 3, colour=[1, 0, 0]
        )


class Strange_r(Particle):
    """
    edge features of a strange quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            m=m_strange,
            S=0.5,
            LIW=-0.5,
            LY=1 / 3,
            RIW=0,
            RY=-2 / 3,
            colour=[1, 0, 0],
        )


class Top_r(Particle):
    """
    edge features of a top quark
    """

    def __init__(self):
        Particle.__init__(
            self, m=m_top, S=0.5, LIW=0.5, LY=1 / 3, RIW=0, RY=4 / 3, colour=[1, 0, 0]
        )


class AntiTop_b(Particle):
    """
    edge features of a down quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            m=m_top,
            S=0.5,
            LIW=0,
            LY=-4 / 3,
            RIW=-0.5,
            RY=-1 / 3,
            anti_colour=[0, 1, 0],
        )


class Bottom_r(Particle):
    """
    edge features of a bottom quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            m=m_bottom,
            S=0.5,
            LIW=-0.5,
            LY=1 / 3,
            RIW=0,
            RY=-2 / 3,
            colour=[1, 0, 0],
        )


## ANCHOR Boson Classes
class Photon(Particle):
    """
    edge features of a photon
    """

    def __init__(self):
        Particle.__init__(self, m=0, S=1, LIW=0, LY=0, RIW=0, RY=0)


class Gluon_rbbar(Particle):
    """
    edge features of a gluon
    """

    def __init__(self):
        Particle.__init__(
            self,
            m=0,
            S=1,
            LIW=0,
            LY=0,
            RIW=0,
            RY=0,
            colour=[1, 0, 0],
            anti_colour=[0, 1, 0],
        )


class W_plus(Particle):
    """
    edge features of W plus boson
    """

    def __init__(self):
        Particle.__init__(self, m=m_W, S=1, LIW=1, LY=0, RIW=0, RY=0)


class W_minus(Particle):
    """
    edge features of W minus boson
    """

    def __init__(self):
        Particle.__init__(self, m=m_W, S=1, LIW=-1, LY=0, RIW=0, RY=0)


class Z_0(Particle):
    """
    edge features of Z boson
    """

    def __init__(self):
        Particle.__init__(self, m=m_Z, S=1, LIW=0, LY=0, RIW=0, RY=0)


class Higgs(Particle):
    """
    edge features of W plus boson
    """

    def __init__(self):
        Particle.__init__(self, m=m_H, S=0, LIW=-0.5, LY=1, RIW=0, RY=0)


test_particle = Mu_plus()
test_particle.print_feat()
