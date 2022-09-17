from dataclasses import dataclass

import numpy as np



q_e   = 1.602e-19        # electron charge in C
h     = 6.626e-34        # Planck constant in Js
hbar  = h /  (2 * np.pi) # Planck constant / 2p in Js
phi_0 = hbar / (2 * q_e) # flux quantum / 2p in Wb



@dataclass
class DeviceParams:
    # Device parameters

    EL:         float # GHz
    EJ:         float # GHz
    EC_phi:     float # GHz
    EC_theta:   float # GHz
   
    def __post_init__(self):
        if self.EL is None:         self.EL = 1.3 
        if self.EJ is None:         self.EJ = 7
        if self.EC_phi is None:     self.EC_phi = 1.2
        if self.EC_theta is None:   self.EC_theta = 0.6


@dataclass
class SimulParams:
    # Simulation parameters

    tol:            float   # diagonalisation tolerance
    keig:           int     # number of required ekets
    N_grid_phi:     int     # number of points in phi direction (odd)
    N_grid_theta:   int     # number of points in theta direction (even)
    D_phi:          float   # maximum range of the phi coordinate
    N_phiext:       int     # number of flux points

    def __post_init__(self):
        if self.tol is None:            self.tol = 1e-10
        if self.keig is None:           self.keig = 50
        if self.N_grid_phi is None:     self.N_grid_phi = 81
        if self.N_grid_theta is None:   self.N_grid_theta = 100
        if self.D_phi is None:          self.D_phi = 6 * np.pi
        if self.N_phiext is None:       self.N_phiext = 21



def get_phiext_grid(N_phiext): 
    return np.linspace(0, np.pi, N_phiext, endpoint=True) # external flux vector