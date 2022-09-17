import numpy as np
import scipy.sparse as ssp
from scipy.sparse.linalg import eigsh
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm


def triflux_operators(D_zeta, D_theta, D_chi, N_zeta, N_theta, N_chi):
    """
    accepts: [D_theta] boundary of the theta grid
             [D_chi] boundary of the chi grid, 
             [N_zeta, N_theta, N_chi] number of points of the zeta, theta, chi grids
    returns: operators for the tri-fluxonium hamiltonion in phaselike basis
    """
    
    # coordinate grids
    # exclude endpoint for compact zeta variable, include endpoint otherwise
    zeta_pts  = np.linspace((-2 * np.pi + D_zeta) * np.sqrt(3), D_zeta * np.sqrt(3), 
                            N_zeta, endpoint=False, dtype='float64') 
    theta_pts = np.linspace(-D_theta, D_theta, N_theta, endpoint=True, dtype='float64')
    chi_pts   = np.linspace(-D_chi, D_chi, N_chi, endpoint=True, dtype='float64') 
    
    # finite element increments
    dzeta     = (zeta_pts[-1] - zeta_pts[0]) / (N_zeta - 1) 
    dtheta    = (theta_pts[-1] - theta_pts[0]) / (N_theta - 1)
    dchi      = (chi_pts[-1] - chi_pts[0]) / (N_chi - 1)
    
    # identity operators
    id_zeta   = ssp.identity(N_zeta, format='csc', dtype='float64')
    id_theta  = ssp.identity(N_theta, format='csc', dtype='float64')
    id_chi    = ssp.identity(N_chi, format='csc', dtype='float64')
    
    # quantity operators
    zeta      = ssp.diags(zeta_pts, 0, shape=(N_zeta,N_zeta), format='csc', dtype='float64')
    theta     = ssp.diags(theta_pts, 0, shape=(N_theta,N_theta), format='csc', dtype='float64')
    chi       = ssp.diags(chi_pts, 0, shape=(N_chi,N_chi), format='csc', dtype='float64')
    
    # quantity squared operators
    theta2    = ssp.diags(theta_pts**2, 0, shape=(N_theta,N_theta), format='csc', dtype='float64')
    chi2      = ssp.diags(chi_pts**2, 0, shape=(N_chi,N_chi), format='csc', dtype='float64')
        
    # trigonometric operators 
    cos_13zeta  = ssp.diags(np.cos(zeta_pts / np.sqrt(3)), 0, shape=(N_zeta,N_zeta), format='csc', dtype='float64')
    sin_13zeta  = ssp.diags(np.sin(zeta_pts / np.sqrt(3)), 0, shape=(N_zeta,N_zeta), format='csc', dtype='float64')
    cos_12theta = ssp.diags(np.cos(theta_pts / np.sqrt(2)), 0, shape=(N_theta,N_theta), format='csc', dtype='float64')
    cos_23chi   = ssp.diags(np.cos(chi_pts * np.sqrt(2/3)), 0, shape=(N_chi,N_chi), format='csc', dtype='float64')
    sin_23chi   = ssp.diags(np.sin(chi_pts * np.sqrt(2/3)), 0, shape=(N_chi,N_chi), format='csc', dtype='float64')
    cos_16chi   = ssp.diags(np.cos(chi_pts / np.sqrt(6)), 0, shape=(N_chi,N_chi), format='csc', dtype='float64')
    sin_16chi   = ssp.diags(np.sin(chi_pts / np.sqrt(6)), 0, shape=(N_chi,N_chi), format='csc', dtype='float64')
    
    # zeta derivatives (compact, with boundary conditions)
    d1_zeta    = (1/2/dzeta) * ssp.diags([-1,1],[-1,1], shape=(N_zeta,N_zeta), format='csc', dtype='float64')
    d1_zeta_bc = (1/2/dzeta) * ssp.diags([1,-1], [-(N_zeta-1),(N_zeta-1)], shape=(N_zeta,N_zeta), 
                                         format='csc', dtype='float64')
    d1_zeta    = d1_zeta + d1_zeta_bc
    d2_zeta    = (1/dzeta**2) * ssp.diags([1,-2,1], [-1,0,1], shape=(N_zeta,N_zeta), format='csc', dtype='float64')
    d2_zeta_bc = (1/dzeta**2) * ssp.diags([1,1], [-(N_zeta-1),(N_zeta-1)], shape=(N_zeta,N_zeta),
                                          format='csc', dtype='float64')
    d2_zeta    = d2_zeta + d2_zeta_bc
    
    # theta derivatives
    d1_theta   = (1/2/dtheta) * ssp.diags([-1,1], [-1,1], shape=(N_theta,N_theta), format='csc', dtype='float64')
    d2_theta   = (1/dtheta**2) * ssp.diags([1,-2,1], [-1,0,1], shape=(N_theta,N_theta), format='csc', dtype='float64')
    
    # chi derivatives
    d1_chi     = (1/2/dchi) * ssp.diags([-1,1], [-1,1], shape=(N_chi,N_chi), format='csc', dtype='float64')
    d2_chi     = (1/dchi**2) * ssp.diags([1,-2,1], [-1,0,1], shape=(N_chi,N_chi), format='csc', dtype='float64')
    
    return [id_zeta,     # identity ops
            id_theta,
            id_chi,
            theta2,      # quantity squared ops
            chi2,
            cos_13zeta,  # trig ops
            sin_13zeta, 
            cos_12theta,
            cos_23chi,
            sin_23chi, 
            cos_16chi, 
            sin_16chi,
            d1_zeta,     # first derivative ops
            d1_theta,
            d1_chi,
            d2_zeta,     # second derivative ops
            d2_theta,
            d2_chi]


def H_triflux_phaselike(phiext, EJ, EC, EL, N_zeta, N_theta, N_chi, D_zeta, D_theta, D_chi):
    """
    accepts: [phiext] phase associated with the external flux (decoupled in this hamiltonian)
             [EJ, EL, EC] circuit parameters
             [N_zeta, N_theta, N_chi] simulation number of points
             [D_zeta, D_theta, D_chi] simuluation upper bound of zeta coordinate, range of theta, chi coordinates
    returns: hamiltonian in phaselike basis
    """

    id_zeta, id_theta, id_chi, theta2, chi2, \
    cos_13zeta, sin_13zeta, cos_12theta, cos_23chi, sin_23chi, cos_16chi, sin_16chi, \
    d1_zeta, d1_theta, d1_chi, \
    d2_zeta, d2_theta, d2_chi \
    = triflux_operators(D_zeta, D_theta, D_chi, N_zeta, N_theta, N_chi)

    return(- 4.0 * EC * ssp.kron(d2_zeta, ssp.kron(id_theta, id_chi)) 
           - 4.0 * EC * ssp.kron(id_zeta, ssp.kron(d2_theta, id_chi))  
           - 4.0 * EC * ssp.kron(id_zeta, ssp.kron(id_theta, d2_chi))  
           - 1.0 * EJ * ssp.kron(cos_13zeta, ssp.kron(id_theta, cos_23chi)) 
           + 1.0 * EJ * ssp.kron(sin_13zeta, ssp.kron(id_theta, sin_23chi))
           - 2.0 * EJ * ssp.kron(cos_13zeta, ssp.kron(cos_12theta, cos_16chi))
           - 2.0 * EJ * ssp.kron(sin_13zeta, ssp.kron(cos_12theta, sin_16chi))
           + 1.5 * EL * ssp.kron(id_zeta, ssp.kron(theta2, id_chi))
           + 1.5 * EL * ssp.kron(id_zeta, ssp.kron(id_theta, chi2))
          )



def diagonalise(operator, keig, ground_guess, tol):
    """
    accepts: [operator] operator to diagonalise
             [keig] number of eigenvectors
             [ground_guess] starting eigenstate for iteration
    returns: eigenvalues and eigenvectors of operator sorted by in increasing energy 
    """

    evals, ekets = ssp.linalg.eigsh(operator, k=keig, which='SA', tol=tol, v0=ground_guess)
    indx         = evals.argsort() # returns the indices that would sort array
    evals_s      = np.sort(evals)
    evals_s      = evals_s - evals_s[0] # sorted and ground normalised to 0
    ekets_s      = ekets[:, indx]
    
    return evals_s, ekets_s



def get_ekets_3D(ekets, simul):
    """
    accepts: [ekets] (N_zeta * N_theta * N_chi) * keig vector to reshape to 3D
             [N_zeta, N_theta, N_chi] simulation number of points
             [keig] number of of eigenvectors
    returns: eigenvector of shape (N_zeta, N_theta, N_chi, keig)
    """

    assert len(ekets.shape) == 2
    assert ekets.shape == (simul.N_zeta * simul.N_theta * simul.N_chi, simul.keig)
    ekets_3D = ekets.reshape(simul.N_zeta, simul.N_theta, simul.N_chi, simul.keig)
    return ekets_3D



def g_calc(evals, ekets, simul):
    """
    evals :  keigs dim array containing eigenvalues. (not really needed)
    ekets :  N_zeta*N_theta*N_chi * keigs dimensional array containing eigenvectors in each column 
    """

    assert simul.keig == len(evals)
    assert ekets.shape == (simul.N_zeta * simul.N_theta * simul.N_chi, simul.keig)

    g_matrix = np.zeros((3, simul.keig, simul.keig))
    
    # only need the first derivative operators
    id_zeta, id_theta, id_chi, theta2, chi2, \
    cos_13zeta, sin_13zeta, cos_12theta, cos_23chi, sin_23chi, cos_16chi, sin_16chi, \
    d1_zeta, d1_theta, d1_chi, \
    d2_zeta, d2_theta, d2_chi \
    = triflux_operators(simul.D_zeta, simul.D_theta, simul.D_chi, simul.N_zeta, simul.N_theta, simul.N_chi)

    oper = [ssp.kron(d1_zeta, ssp.kron(id_theta, id_chi)),
            ssp.kron(id_zeta, ssp.kron(d1_theta, id_chi)),
            ssp.kron(id_zeta, ssp.kron(id_theta, d1_chi))]

    for op in np.arange(3):
        for i in np.arange(simul.keig):
            for j in np.arange(simul.keig):
                g_matrix[op,i,j] = np.vdot(ekets[:,i], oper[op].dot(ekets[:,j]))
    
    return g_matrix



def write2vtk(matrix, fname="tree_flux.vtk"):
    """
    write matrix to fname in vtk format to visualise
    """
    sx, sy, sz = matrix.shape
    norm = np.linalg.norm(matrix)
    lines ='# vtk DataFile Version 2.0\nVolume example\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\nASPECT_RATIO 1 1 1\nORIGIN 0 0 0\nPOINT_DATA %d\nSCALARS matlab_scalars float 1\nLOOKUP_TABLE default\n'%(sx, sy, sz, matrix.size)
    with open(fname, 'w') as f:
        f.write(lines)
        for ix in range(sz):
            v = np.ravel(matrix[:,:,ix], order="f")
            v = ["%1.5f"%x for x in (100 * v /norm)]
            line = " ".join(v)
            f.write(line+"\n")


def log_me(device, simul, elapsed_time, evals, fname="log.txt"):
    """
    save log of simulation run
    device: parameters of trifluxonium
    simul: finite element discretisation settings
    evals: energy eigenvalues returned by diagonalisation
    fname: name of log file
    """

    f = open(fname, "w")
    f.write("\nSIMULATION LOG\n")
    f.write("{}\n\n".format(datetime.datetime.now()))
        
    f.write("tol = {}\n".format(simul.tol) + 
            "keig = {}\n".format(simul.keig) + 
            "N_zeta = {}\n".format(simul.N_zeta) +
            "N_theta = {}\n".format(simul.N_theta) +
            "N_chi = {}\n".format(simul.N_chi) +
            "D_zeta = {}\n".format(simul.D_zeta) +
            "D_theta = {}\n".format(simul.D_theta) +
            "D_chi = {}\n".format(simul.D_chi)
        )

    f.write("\n")
    f.write("EC = {}\n".format(device.EC) +
            "EJ = {}\n".format(device.EJ) + 
            "EL = {}\n".format(device.EL)
        )

    f.write("\n")
    f.write("elapsed time: {}\n".format(elapsed_time))

    f.write("\n")
    f.write("energies\n")
    for i in range(len(evals)):
        f.write("{}\n".format(evals[i]))
    f.close()

