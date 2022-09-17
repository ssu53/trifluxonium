from dataclasses import dataclass

import numpy as np



q_e   = 1.602e-19        # electron charge in C
h     = 6.626e-34        # Planck constant in Js
hbar  = h /  (2 * np.pi) # Planck constant / 2p in Js
phi_0 = hbar / (2 * q_e) # flux quantum / 2p in Wb



@dataclass
class DeviceParams:
    # Device parameters

    EC:         float # GHz. Small EC ~ large mass.
    EJ:         float # GHz. Small EJ ~ small cosine fluctations.
    EL:         float # GHz. Small EL ~ shallow well.
   
    def __post_init__(self):
        if self.EC is None:         self.EC = 2
        if self.EJ is None:         self.EJ = 15
        if self.EL is None:         self.EL = 0.3



@dataclass
class SimulParams:
    # Simulation parameters
    # State is described by: zeta (compact), theta, chi

    tol:            float   # diagonalisation tolerance
    keig:           int     # number of required ekets

    N_zeta:         int     # number of points in zeta direction (aesthetically even because compact)
    N_theta:        int     # number of points in theta direction (aesthetically odd, because not compact)
    N_chi:          int     # number of points in chi direction (aesthetically odd, because not compact)
    
    D_zeta:         float   # upper bound of (compact) zeta coordinate (here: symmetric -pi to pi choice)
    D_theta:        float   # maximum range of (non compact) theta coordinate
    D_chi:          float   # maximum range of (non compact) chi coordinate

    def __post_init__(self):
        if self.tol is None:            self.tol = 1e-10
        if self.keig is None:           self.keig = 10
        if self.N_zeta is None:         self.N_zeta = 100
        if self.N_theta is None:        self.N_theta = 101
        if self.N_chi is None:          self.N_chi = 101
        if self.D_zeta is None:         self.D_zeta = np.pi
        if self.D_theta is None:        self.D_theta = 4 * np.pi
        if self.D_chi is None:          self.D_chi = 4 * np.pi



def get_phiext_grid(N_phiext=21): 
    """
    will become relevant to analysis if considering different flux through the loops
    N_phiext: number of flux points
    """
    return np.linspace(0, np.pi, N_phiext, endpoint=True) # external flux vector