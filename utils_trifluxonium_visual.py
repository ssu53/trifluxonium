import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_eigenstates(evals, ekets_3D, simul, dim_trace_over=0, figsize=(12,5), fname=None):
    """
    accepts: [evals] eigenvalues
             [ekets_3D] eigenstates
             [simul] simulation settings
             [trace_over] dimension (0, 1, 2 -> zeta, theta, chi respecitvely) to trace over
             [figsize] figure size
             [fname] filename to which to save figure; not saved if None
    returns: figure
    """

    assert simul.keig == len(evals)
    assert ekets_3D.shape == (simul.N_zeta, simul.N_theta, simul.N_chi, simul.keig)

    if dim_trace_over == 0:
        title = "Eigenstates as function of theta, chi. Mean over zeta."
        subplot_layout = (2, int(np.ceil(simul.keig/2)))
        extent = [-simul.D_chi, simul.D_chi, -simul.D_theta, simul.D_theta]
        xlabel = r"${\chi}$"
        ylabel = r"${\theta}$"
        xticks = [-6*np.pi, -4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi, 6*np.pi]
        xticklabels = [r"$-6{\pi}$", r"$-4{\pi}$", r"$-2{\pi}$", 0, r"$2{\pi}$", r"$4{\pi}$", r"$6{\pi}$"]
        yticks = xticks
        yticklabels = xticklabels
    
    elif dim_trace_over == 1:
        title = "Eigenstates as function of zeta, chi. Mean over theta."
        subplot_layout = (int(np.ceil(simul.keig/4)), 4)
        extent = [-simul.D_chi, simul.D_chi, -2*np.pi+simul.D_zeta, simul.D_zeta]
        xlabel = r"${\chi}$"
        ylabel = r"${\zeta}$"
        xticks = [-6*np.pi, -4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi, 6*np.pi]
        xticklabels = [r"$-6{\pi}$", r"$-4{\pi}$", r"$-2{\pi}$", 0, r"$2{\pi}$", r"$4{\pi}$", r"$6{\pi}$"]
        yticks = [-np.pi, np.pi]
        yticklabels = [r"$-{\pi}$", r"${\pi}$"]
    
    elif dim_trace_over == 2:
        title = "Eigenstates as function of zeta, theta. Mean over chi."
        subplot_layout = (int(np.ceil(simul.keig/4)), 4)
        extent = [-simul.D_theta, simul.D_theta, -2*np.pi+simul.D_zeta, simul.D_zeta]
        xlabel = r"${\theta}$"
        ylabel = r"${\zeta}$"
        xticks = [-6*np.pi, -4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi, 6*np.pi]
        xticklabels = [r"$-6{\pi}$", r"$-4{\pi}$", r"$-2{\pi}$", 0, r"$2{\pi}$", r"$4{\pi}$", r"$6{\pi}$"]
        yticks = [-np.pi, np.pi]
        yticklabels = [r"$-{\pi}$", r"${\pi}$"]
    
    else:
        raise Exception("Invalid dimension to trace over.")
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=14)

    for n_level in range(0, simul.keig):

        ax = fig.add_subplot(*subplot_layout, n_level+1)
        image = np.mean(ekets_3D, dim_trace_over)[:,:,n_level]
        vbound = np.max(np.abs(image))
        
        ax.imshow(
            image, 
            cmap='RdBu', 
            extent=extent, 
            vmin=-vbound,vmax=vbound)
        
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title('$E_%s$ = %s GHz' %(n_level, np.round(evals[n_level],3)), fontsize=9)
        ax.axis('scaled')
        plt.tight_layout()

    if fname is not None: plt.savefig(fname)

    return plt



def get_coordinates(simul):
    """
    accepts: [simul] simulation settings
    returns: coordinates in zeta, theta, chi dimensions respectively
    """

    zeta_pts  = np.linspace(-2 * np.pi + simul.D_zeta, simul.D_zeta, simul.N_zeta, endpoint=False)
    theta_pts = np.linspace(-simul.D_theta, simul.D_theta, simul.N_theta, endpoint=True)
    chi_pts   = np.linspace(-simul.D_chi, simul.D_chi, simul.N_chi, endpoint=True)

    return zeta_pts, theta_pts, chi_pts


def potential(device, simul):
    """
    accepts: [device] device settings
             [simul] simulation settings
    returns: potential as function of discretised zeta, theta, chi respectively; shape (N_zeta, N_theta, N_chi)
    """

    print(f"Computing potential for trifluxonium given by {device}.")

    zeta_pts, theta_pts, chi_pts = get_coordinates(simul)
    zeta_pts  = zeta_pts.reshape(-1,1,1)
    theta_pts = theta_pts.reshape(1,-1,1)
    chi_pts   = chi_pts.reshape(1,1,-1)

    V_3D = - device.EJ * (np.cos((zeta_pts + np.sqrt(2) * chi_pts) / np.sqrt(3)) + 2 * np.cos(theta_pts / np.sqrt(2)) * 
                          np.cos(zeta_pts / np.sqrt(3) - chi_pts / np.sqrt(6))) + 1.5 * device.EL * (theta_pts**2 + chi_pts**2)

    return V_3D


def plot_potential(index_to_slice, dim_to_slice, V_3D, simul, show_colorbar=False, figsize=(12,8), fname=None):
    """
    accepts: [slice] slice of the discretisation to visualise in the dim_to_slice dimension
             [dim_to_slice] dimension (0, 1, 2 -> zeta, theta, chi respecitvely) along which to slice
             [simul] simulation settings
             [figsize] figure size
             [fname] filename to which to save figure; not saved if None
    returns: figure
    """

    assert V_3D.shape == (simul.N_zeta, simul.N_theta, simul.N_chi)

    zeta_pts, theta_pts, chi_pts = get_coordinates(simul)

    if dim_to_slice == 0:
        assert 0 <= index_to_slice < simul.N_zeta
        title = f"Energy potential as function of theta and chi, at zeta = {zeta_pts[index_to_slice]/np.pi:.2f}${{\pi}}$."
        X, Y = np.meshgrid(theta_pts, chi_pts)
        Z = V_3D[index_to_slice,:,:].T
        xlabel = r"${\theta}$"
        ylabel = r"${\chi}$"
        xticks = ([-4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi],[r"$-4{\pi}$", r"$-2{\pi}$", 0, r"$2{\pi}$", r"$4{\pi}$"])
        yticks = ([-4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi],[r"$-4{\pi}$", r"$-2{\pi}$", 0, r"$2{\pi}$", r"$4{\pi}$"])

    elif dim_to_slice == 1:
        assert 0 <= index_to_slice < simul.N_theta
        title = f"Energy potential as function of zeta and chi, at theta = {theta_pts[index_to_slice]/np.pi:.2f}${{\pi}}$."
        X, Y = np.meshgrid(zeta_pts, chi_pts)
        Z = V_3D[:,index_to_slice,:].T
        xlabel = r"${\zeta}$"
        ylabel = r"${\chi}$"
        xticks = ([-np.pi, 0, np.pi], [r"$-{\pi}$", "0", r"${\pi}$"])
        yticks = ([-4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi],[r"$-4{\pi}$", r"$-2{\pi}$", 0, r"$2{\pi}$", r"$4{\pi}$"])

    elif dim_to_slice == 2:
        assert 0 <= index_to_slice < simul.N_chi
        title = f"Energy potential as function of zeta and theta, at chi = {chi_pts[index_to_slice]/np.pi:.2f}${{\pi}}$."
        X, Y = np.meshgrid(zeta_pts, theta_pts)
        Z = V_3D[:,:,index_to_slice].T
        xlabel = r"${\zeta}$"
        ylabel = r"${\theta}$"
        xticks = ([-np.pi, 0, np.pi], [r"$-{\pi}$", "0", r"${\pi}$"])
        yticks = ([-4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi],[r"$-4{\pi}$", r"$-2{\pi}$", 0, r"$2{\pi}$", r"$4{\pi}$"])

    else: 
        raise Exception("Invalid dimension to slice.")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.view_init(elev=30, azim=-30)
    if show_colorbar: fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlabel('potential (GHz)', fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xticks(*xticks)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_yticks(*yticks)
    ax.set_title(title, fontsize=14)

    if fname is not None: plt.savefig(fname)

    return fig

