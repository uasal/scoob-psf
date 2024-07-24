import numpy as np
import astropy.units as u
from astropy.io import fits

import os
from pathlib import Path
import time
import copy

import poppy

from scipy.optimize import minimize

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))
from . import utils
from . import props
from .math_module import xp, _scipy, ensure_np_array
from .imshows import imshow1, imshow2, imshow3

def create_poke_modes(m):
    poke_modes = xp.zeros((m.Nacts, m.Nact, m.Nact))
    count = 0
    for i in range(m.Nact):
        for j in range(m.Nact):
            if m.dm_mask[i,j]:
                poke_modes[count, i,j] = 1
                count += 1
    
    return poke_modes

def compute_jacobian(m,
                     modes,
                     amp=1e-9):
    Nmodes = modes.shape[0]
    jac = xp.zeros((2*m.Nmask, Nmodes))
    for i in range(Nmodes):
        E_pos = m.forward(amp*modes[i][m.dm_mask], use_wfe=True, use_vortex=True)[m.control_mask]
        E_neg = m.forward(-amp*modes[i][m.dm_mask], use_wfe=True, use_vortex=True)[m.control_mask]
        response = (E_pos - E_neg)/(2*amp)
        jac[::2,i] = xp.real(response)
        jac[1::2,i] = xp.imag(response)

    return jac

def beta_reg(S, beta=-1):
    # S is the sensitivity matrix also known as the Jacobian
    sts = xp.matmul(S.T, S)
    rho = xp.diag(sts)
    alpha2 = rho.max()

    control_matrix = xp.matmul( xp.linalg.inv( sts + alpha2*10.0**(beta)*xp.eye(sts.shape[0]) ), S.T)
    return control_matrix

def efc(m,
        control_matrix,  
        Nitr=3, 
        nominal_command=None, 
        gain=0.5, 
        all_ims=None, 
        all_efs=None,
        all_commands=None,
        ):
    
    metric_images = [] if all_ims is None else all_ims
    ef_estimates = [] if all_efs is None else all_efs
    dm_commands = [] if all_commands is None else all_commands
    starting_itr = len(metric_images)

    total_command = copy.copy(nominal_command) if nominal_command is not None else xp.zeros((m.Nact,m.Nact))
    del_command = xp.zeros((m.Nact,m.Nact))
    E_ab = xp.zeros(2*m.Nmask)
    for i in range(Nitr):
        E_est = m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True,)
        E_ab[::2] = E_est.real[m.control_mask]
        E_ab[1::2] = E_est.imag[m.control_mask]

        del_acts = -gain * control_matrix.dot(E_ab)
        del_command[m.dm_mask] = del_acts
        total_command += del_command
        
        image_ni = xp.abs(m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True))**2

        mean_ni = xp.mean(image_ni[m.control_mask])

        metric_images.append(copy.copy(image_ni))
        ef_estimates.append(copy.copy(E_ab))
        dm_commands.append(copy.copy(total_command))

        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=m.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-9)

    return metric_images, ef_estimates, dm_commands

def sim(m, val_and_grad,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        all_ims=[], 
        all_efs=[],
        all_commands=[],
        ):
    """_summary_

    Parameters
    ----------
    m : object
        The model of the coronagraph
    val_and_grad : function
        The function used to compute the cost-function value and the 
        gradient with respect to DM actuators
    Nitr : int, optional
        Number of EFC iterations, by default 3
    nominal_command : np.ndarray, optional
        the initial command to begin with, by default None
    reg_cond : float, optional
        Tikhonov regularization parameter being used for EFC, by default 1e-2
    bfgs_tol : float, optional
        BFGS tolerance, by default 1e-3
    bfgs_opts : dict, optional
        Dictionary of options being used for BFGS, by default None
    gain : float, optional
        _description_, by default 0.5
    all_ims : list, optional
        All commands saved from previous iterations, by default None
    all_efs : list, optional
        All commands saved from previous iterations, by default None
    all_commands : list, optional
        All commands saved from previous iterations, by default None

    Returns
    -------
    _type_
        _description_
    """
    
    starting_itr = len(all_ims)

    # E_model_nom = ensure_np_array(m.forward(np.zeros(m.Nacts), use_vortex=True, use_wfe=False) * m.control_mask)
    if len(all_commands)>0:
        total_command = copy.copy(all_commands[-1])
    else:
        total_command = xp.zeros((m.Nact,m.Nact))
    del_command = xp.zeros((m.Nact,m.Nact))
    del_acts0 = np.zeros(m.Nacts)
    for i in range(Nitr):
        E_ab = m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True,)

        res = minimize(val_and_grad, 
                       jac=True, 
                       x0=del_acts0,
                    #    args=(m, E_ab, reg_cond, E_target, E_model_nom), 
                       args=(m, ensure_np_array(total_command[m.dm_mask]), E_ab, reg_cond), 
                       method='L-BFGS-B',
                       tol=bfgs_tol,
                       options=bfgs_opts,
                       )

        del_acts = gain * res.x
        del_command[m.dm_mask] = del_acts
        total_command += del_command
        image_ni = xp.abs(m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True))**2

        mean_ni = xp.mean(image_ni[m.control_mask])

        all_ims.append(copy.copy(image_ni))
        all_efs.append(copy.copy(E_ab))
        all_commands.append(copy.copy(total_command))

        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=m.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-9)

    return all_ims, all_efs, all_commands


def run(sysi, 
        m, val_and_grad,
        est_fun, est_params,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        all_ims=[], 
        all_efs=[],
        all_commands=[],
        ):
    
    starting_itr = len(all_ims)

    # E_model_nom = ensure_np_array(m.forward(np.zeros(m.Nacts), use_vortex=True, use_wfe=False) * m.control_mask)
    if len(all_commands)>0:
        total_command = copy.copy(all_commands[-1])
    else:
        total_command = xp.zeros((m.Nact,m.Nact))
    del_command = xp.zeros((m.Nact,m.Nact))
    del_acts0 = np.zeros(m.Nacts)
    for i in range(Nitr):
        E_ab = est_fun(sysi, est_params)
        
        res = minimize(val_and_grad, 
                       jac=True, 
                       x0=del_acts0,
                    #    args=(m, E_ab, reg_cond, E_target, E_model_nom), 
                       args=(m, ensure_np_array(total_command[m.dm_mask]), E_ab, reg_cond), 
                       method='L-BFGS-B',
                       tol=bfgs_tol,
                       options=bfgs_opts,
                       )

        del_acts = gain * res.x
        del_command[m.dm_mask] = del_acts
        total_command += del_command

        # sysi.set_dm(total_command)
        sysi.add_dm(del_command)
        image_ni = sysi.snap()

        mean_ni = xp.mean(image_ni[m.control_mask])

        all_ims.append(copy.copy(image_ni))
        all_efs.append(copy.copy(E_ab))
        all_commands.append(copy.copy(total_command))

        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=m.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-9)

    return all_ims, all_efs, all_commands


