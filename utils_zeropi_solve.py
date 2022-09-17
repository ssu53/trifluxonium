import numpy as np
import scipy.sparse as ssp
from scipy.sparse.linalg import eigsh
import time


def zeropi_operators(D_phi, N_phi, N_theta):
    """
    accepts: [D_phi] boundary of the phi grid, [N_phi, N_theta] number of points of the phi and theta grid
    returns: operators for the zero-pi Hamiltonian in phaselike basis
    """
    
    phi_pts     = np.linspace(-D_phi, D_phi, N_phi, endpoint=True, dtype='float64')
    theta_pts   = np.linspace(- 0.5 * np.pi, 1.5 * np.pi, N_theta, endpoint=False, dtype='float64')
    dp          = (phi_pts[-1] - phi_pts[0]) / (N_phi - 1)
    dt          = (theta_pts[-1] - theta_pts[0]) / (N_theta - 1) 
    
    id_phi      = ssp.identity(N_phi, format='csc', dtype='float64')
    id_theta    = ssp.identity(N_theta, format='csc', dtype='float64')
    phi         = ssp.diags(phi_pts, 0, shape=(N_phi,N_phi), format='csc', dtype='float64')
    phi2        = ssp.diags(phi_pts**2, 0, shape=(N_phi,N_phi), format='csc', dtype='float64')
    cos_phi     = ssp.diags(np.cos(phi_pts), 0, shape=(N_phi,N_phi), format='csc', dtype='float64')
    sin_phi     = ssp.diags(np.sin(phi_pts), 0, shape=(N_phi,N_phi), format='csc', dtype='float64')
    cos_theta   = ssp.diags(np.cos(theta_pts), 0, shape=(N_theta,N_theta), format='csc', dtype='float64')
    d1_phi      = (1/2/dp) * ssp.diags([-1,1], [-1,1], shape=(N_phi,N_phi), format='csc', dtype='float64')
    d2_phi      = (1/dp**2) * ssp.diags([1,-2,1], [-1,0,1], shape=(N_phi,N_phi), format='csc', dtype='float64') 
    d1_theta    = (1/2/dt) * ssp.diags([-1,1], [-1,1], shape=(N_theta,N_theta), format='csc', dtype='float64')
    d1_theta_bc = (1/2/dt) * ssp.diags([1,-1], [-(N_theta-1),(N_theta-1)], shape=(N_theta,N_theta), format='csc', dtype='float64')
    d1_theta    = d1_theta + d1_theta_bc
    d2_theta    = (1/dt**2) * ssp.diags([1,-2,1], [-1,0,1], shape=(N_theta,N_theta), format='csc', dtype='float64') 
    d2_theta_bc = (1/dt**2) * ssp.diags([1,1], [-(N_theta-1),(N_theta-1)], shape=(N_theta,N_theta), format='csc', dtype='float64') 
    d2_theta    = d2_theta + d2_theta_bc
    
    return [id_phi, id_theta, phi, phi2, cos_phi, sin_phi, cos_theta, d1_phi, d2_phi, d1_theta, d2_theta]

def H_zeropi_phaselike(phiext, EC_phi, EC_theta, EJ, EL, D_phi, N_grid_phi, N_grid_theta):
    """
    accepts: [phiext] phase associated to the external flux, [EC_phi, EC_theta, EJ, EL] zeropi parameters
    returns: zero-pi Hamiltonian in phaselike basis
    """
    
    id_phi, id_theta, phi, phi2, cos_phi, sin_phi, cos_theta, d1_phi, d2_phi, d1_theta, d2_theta = zeropi_operators(D_phi, N_grid_phi, N_grid_theta) 

    return (- 2.0 * EC_phi * ssp.kron(d2_phi, id_theta) +
            - 2.0 * EC_theta * ssp.kron(id_phi, d2_theta) +
            - 2.0 * EJ * ssp.kron(id_phi, cos_theta) * ssp.kron(cos_phi, id_theta) * np.cos(phiext / 2) +
            - 2.0 * EJ * ssp.kron(id_phi, cos_theta) * ssp.kron(sin_phi, id_theta) * np.sin(phiext / 2) +
            + 1.0 * EL * ssp.kron(phi2, id_theta) )


def diagonalize(operator, number_of_ekets, ground_guess, tol):
    """
    returns [evals, ekets]: eigenenergies and eigenvectors sorted in energy 
    """
    evals, ekets = eigsh(operator,k=number_of_ekets,which='SA',tol=tol,v0=ground_guess)
    indx         = evals.argsort()
    evals_s      = np.sort(evals)
    evals_s      = evals_s - evals_s[0]
    ekets_s      = np.zeros(ekets.shape, dtype='float64')
    for i in range(number_of_ekets):
        ekets_s[:,i] = ekets[:,indx[i]]
    return evals_s, ekets_s


def sweep_diagonalize_and_sort_by_overlap(parameteric_operator, number_of_ekets, sweep_vector, N_grid_phi, N_grid_theta, tol):
    """
    returns [evals, ekets, ekets_2D]: eigenenergies and eigenvectors sorted by overlap
    """
    Haux        = parameteric_operator(sweep_vector[0]).toarray()
    sweep_evals = np.zeros((sweep_vector.size,number_of_ekets), dtype='float64')
    sweep_ekets = np.zeros((sweep_vector.size,Haux.shape[0],number_of_ekets), dtype='float64')
    sweep_ekets_2D = np.zeros((sweep_vector.size,N_grid_phi,N_grid_theta,number_of_ekets), dtype='float64')
    k           = 0
    t_av        = 0.0
    v0          = np.random.rand(Haux.shape[0])
    for element in sweep_vector:
        if (k == 0):
            start              = time.time()
            evals, ekets       = diagonalize(parameteric_operator(element), number_of_ekets, v0, tol)
            end                = time.time()
            t_av               = (k * t_av + (end-start))/(k + 1)
            sweep_evals[k,:]   = evals
            sweep_ekets[k,:,:] = ekets
            ekets_prev         = ekets
            v0                 = ekets[:,0]
            k                 += 1
            print('%d%% completed. Remaining time ~ %d min' % (int(k/sweep_vector.size * 100), int((sweep_vector.size - k) * t_av / 60.0)), end='\r')
        else:
            start                  = time.time()
            evals, ekets           = diagonalize(parameteric_operator(element), number_of_ekets, v0, tol)
            indx_sort_by_character = np.zeros(number_of_ekets,dtype='int32')
            for i in range(number_of_ekets):
                ei                = ekets[:,i]
                overlap_with_prev = np.zeros(number_of_ekets,dtype='float64')
                for j in range(number_of_ekets):
                    ej_prev              = ekets_prev[:,j]
                    overlap_with_prev[j] = np.absolute(np.vdot(ei,ej_prev))
                jmax                            = np.argmax(overlap_with_prev)
                indx_sort_by_character[i]       = jmax
                sweep_evals[k,jmax]   = evals[i]
                sweep_ekets[k,:,jmax] = ekets[:,i]
                ekets_prev[:,jmax]    = ekets[:,i]
            end  = time.time()
            t_av = (k * t_av + (end-start))/(k + 1)
            v0 = ekets[:,0]
            k += 1
            print('%d%% completed. Remaining time ~ %d min' % (int(k/sweep_vector.size * 100), int((sweep_vector.size - k) * t_av / 60.0)), end='\r')
     
    for flux_idx in range(sweep_vector.size):
        for num_idx in range(number_of_ekets):
            sweep_ekets_2D[flux_idx,:,:,num_idx] = sweep_ekets[flux_idx,:,num_idx].reshape(N_grid_phi, N_grid_theta)
    
    return sweep_evals, sweep_ekets, sweep_ekets_2D


def parametric_H(phiext, EC_phi, EC_theta, EJ, EL, D_phi, N_grid_phi, N_grid_theta):
    return H_zeropi_phaselike(phiext, EC_phi, EC_theta, EJ, EL, D_phi, N_grid_phi, N_grid_theta)

