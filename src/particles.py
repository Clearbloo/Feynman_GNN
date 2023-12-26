import numpy as np
from typing import List, Type

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
    """
    Access all particles
    """

    _registry = {}

    @classmethod
    def register_particle(cls, particle_class: Type["Particle"]):
        cls._registry[particle_class.__name__.lower()] = particle_class

    @classmethod
    def get_particle_class(cls, name: str) -> Type["Particle"]:
        return cls._registry.get(name.lower())

    @classmethod
    def list_particles(cls) -> List[str]:
        return list(cls._registry.keys())


class Particle:
    """
    edge features vector
    l=[m,S,LIW3,LY,RIW3,RY,colour,h]

    Represents a particle with various properties.
    Masses are given in MeV.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        ParticleRegistry.register_particle(cls)

    def __init__(
        self,
        mass: float,
        spin: float,
        left_weak_isospin: float,
        left_hypercharge: float,
        right_weak_isospin: float,
        right_hypercharge: float,
        colour: List[int] = [0, 0, 0],
        anti_colour: List[int] = [0, 0, 0],
    ):
        self.validate_colour(colour)
        self.validate_colour(anti_colour)

        self.mass = mass
        self.spin = spin
        self.left_weak_isospin = left_weak_isospin
        self.left_hypercharge = left_hypercharge
        self.right_weak_isospin = right_weak_isospin
        self.right_hypercharge = right_hypercharge
        self.colour = colour
        self.anti_colour = anti_colour

        self.features = [
            mass,
            spin,
            left_weak_isospin,
            left_hypercharge,
            right_weak_isospin,
            right_hypercharge,
            *colour,
            *anti_colour,
        ]

    @property
    def id(self) -> str:
        return self.__class__.__name__.lower()

    def validate_colour(self, colour):
        if len(colour) != 3 or not all(isinstance(c, int) for c in colour):
            raise ValueError("Colour must be a list of three integers.")

    def get_features(self):
        return self.features

    def anti_particle(self):
        self.features[2], self.features[4] = -self.features[4], -self.features[2]
        self.features[3], self.features[5] = -self.features[5], -self.features[3]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mass={self.mass}, spin={self.spin}, "
            f"left_weak_isospin={self.left_weak_isospin}, left_hypercharge={self.left_hypercharge}, "
            f"right_weak_isospin={self.right_weak_isospin}, right_hypercharge={self.right_hypercharge}, "
            f"colour={self.colour}, anti_colour={self.anti_colour})"
        )

    def print_features(self):
        print(self.features)


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
        Particle.__init__(self, mass=m_e, spin=0.5, left_weak_isospin=-0.5, left_hypercharge=-1, right_weak_isospin=0, right_hypercharge=-2)


class E_plus(Particle):
    """
    Class to construct the edge feautres of a positron
    """

    def __init__(self):
        """
        h = helicity
        p = 3-momentum vector
        """
        Particle.__init__(self, mass=m_e, spin=0.5, left_weak_isospin=0, left_hypercharge=2, right_weak_isospin=0.5, right_hypercharge=1)


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
        Particle.__init__(self, mass=m_mu, spin=0.5, left_weak_isospin=-0.5, left_hypercharge=-1, right_weak_isospin=0, right_hypercharge=-2)


class Mu_plus(Particle):
    """
    Class to construct the edge feautres of an anti-muon
    """

    def __init__(self):
        """
        h = helicity
        p = 3-momentum vector
        """
        Particle.__init__(self, mass=m_mu, spin=0.5, left_weak_isospin=0, left_hypercharge=2, right_weak_isospin=0.5, right_hypercharge=1)


## ANCHOR Quark Classes
class Up_r(Particle):
    """
    edge features of an up quark
    """

    def __init__(self):
        Particle.__init__(
            self, mass=m_up, spin=0.5, left_weak_isospin=0.5, left_hypercharge=1 / 3, right_weak_isospin=0, right_hypercharge=4 / 3, colour=[1, 0, 0]
        )


class AntiUp_b(Particle):
    """
    edge features of an up quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            mass=m_up,
            spin=0.5,
            left_weak_isospin=0,
            left_hypercharge=-4 / 3,
            right_weak_isospin=-0.5,
            right_hypercharge=-1 / 3,
            anti_colour=[0, 1, 0],
        )


class Down_r(Particle):
    """
    edge features of a down quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            mass=m_down,
            spin=0.5,
            left_weak_isospin=-0.5,
            left_hypercharge=1 / 3,
            right_weak_isospin=0,
            right_hypercharge=-2 / 3,
            colour=[1, 0, 0],
        )


class Charm_r(Particle):
    """
    edge features of a charm quark
    """

    def __init__(self):
        Particle.__init__(
            self, mass=m_charm, spin=0.5, left_weak_isospin=0.5, left_hypercharge=1 / 3, right_weak_isospin=0, right_hypercharge=4 / 3, colour=[1, 0, 0]
        )


class Strange_r(Particle):
    """
    edge features of a strange quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            mass=m_strange,
            spin=0.5,
            left_weak_isospin=-0.5,
            left_hypercharge=1 / 3,
            right_weak_isospin=0,
            right_hypercharge=-2 / 3,
            colour=[1, 0, 0],
        )


class Top_r(Particle):
    """
    edge features of a top quark
    """

    def __init__(self):
        Particle.__init__(
            self, mass=m_top, spin=0.5, left_weak_isospin=0.5, left_hypercharge=1 / 3, right_weak_isospin=0, right_hypercharge=4 / 3, colour=[1, 0, 0]
        )


class AntiTop_b(Particle):
    """
    edge features of a down quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            mass=m_top,
            spin=0.5,
            left_weak_isospin=0,
            left_hypercharge=-4 / 3,
            right_weak_isospin=-0.5,
            right_hypercharge=-1 / 3,
            anti_colour=[0, 1, 0],
        )


class Bottom_r(Particle):
    """
    edge features of a bottom quark
    """

    def __init__(self):
        Particle.__init__(
            self,
            mass=m_bottom,
            spin=0.5,
            left_weak_isospin=-0.5,
            left_hypercharge=1 / 3,
            right_weak_isospin=0,
            right_hypercharge=-2 / 3,
            colour=[1, 0, 0],
        )


## ANCHOR Boson Classes
class Photon(Particle):
    """
    edge features of a photon
    """

    def __init__(self):
        Particle.__init__(self, mass=0, spin=1, left_weak_isospin=0, left_hypercharge=0, right_weak_isospin=0, right_hypercharge=0)


class Gluon_rbbar(Particle):
    """
    edge features of a gluon
    """

    def __init__(self):
        Particle.__init__(
            self,
            mass=0,
            spin=1,
            left_weak_isospin=0,
            left_hypercharge=0,
            right_weak_isospin=0,
            right_hypercharge=0,
            colour=[1, 0, 0],
            anti_colour=[0, 1, 0],
        )


class W_plus(Particle):
    """
    edge features of W plus boson
    """

    def __init__(self):
        Particle.__init__(self, mass=m_W, spin=1, left_weak_isospin=1, left_hypercharge=0, right_weak_isospin=0, right_hypercharge=0)


class W_minus(Particle):
    """
    edge features of W minus boson
    """

    def __init__(self):
        Particle.__init__(self, mass=m_W, spin=1, left_weak_isospin=-1, left_hypercharge=0, right_weak_isospin=0, right_hypercharge=0)


class Z_0(Particle):
    """
    edge features of Z boson
    """

    def __init__(self):
        Particle.__init__(self, mass=m_Z, spin=1, left_weak_isospin=0, left_hypercharge=0, right_weak_isospin=0, right_hypercharge=0)


class Higgs(Particle):
    """
    edge features of W plus boson
    """

    def __init__(self):
        Particle.__init__(self, mass=m_H, spin=0, left_weak_isospin=-0.5, left_hypercharge=1, right_weak_isospin=0, right_hypercharge=0)
