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

def sim(m, val_and_grad,
        Nitr=3, 
        nominal_command=None, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        E_target=0.0,
        all_ims=None, 
        all_efs=None,
        all_commands=None,
        ):
    """_summary_

    Parameters
    ----------
    m : object
        The model of the coronagraph
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
    all_ims : _type_, optional
        _description_, by default None
    all_efs : _type_, optional
        _description_, by default None
    all_commands : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    
    metric_images = [] if all_ims is None else all_ims
    ef_estimates = [] if all_efs is None else all_efs
    dm_commands = [] if all_commands is None else all_commands
    starting_itr = len(metric_images)

    E_model_nom = ensure_np_array(m.forward(np.zeros(m.Nacts), use_vortex=True, use_wfe=False) * m.control_mask)

    total_command = copy.copy(nominal_command) if nominal_command is not None else xp.zeros((m.Nact,m.Nact))
    del_command = xp.zeros((m.Nact,m.Nact))
    E_ab = xp.zeros(2*m.Nmask)
    for i in range(Nitr):
        E_ab = m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True,)

        del_acts0 = np.zeros(m.Nacts)
        res = minimize(val_and_grad, 
                       jac=True, 
                       x0=del_acts0,
                       args=(m, E_ab, reg_cond, E_target, E_model_nom), 
                       method='L-BFGS-B',
                       tol=bfgs_tol,
                       options=bfgs_opts,
                       )

        del_acts = gain * res.x
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


def run(sysi, 
        m, 
        est_fun, est_params,
        Nitr=3, 
        nominal_command=None, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        all_ims=None, 
        all_efs=None,
        all_commands=None,
        ):
    """_summary_

    Parameters
    ----------
    sysi : object
        The system interface being used for the instrument
    am : object
        The adjoint model of the coronagraph
    est_fun : function
        The estimation function used to compute the electric field estimate
    est_params : dict
        Dictionary of parameters required for the estimation function
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
    all_ims : _type_, optional
        _description_, by default None
    all_efs : _type_, optional
        _description_, by default None
    all_commands : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    
    metric_images = [] if all_ims is None else all_ims
    ef_estimates = [] if all_efs is None else all_efs
    dm_commands = [] if all_commands is None else all_commands
    starting_itr = len(metric_images)

    total_command = copy.copy(nominal_command) if nominal_command is not None else xp.zeros((m.Nact,m.Nact))
    del_command = xp.zeros((m.Nact,m.Nact))
    E_ab = xp.zeros(2*m.Nmask)
    for i in range(Nitr):
        E_ab = est_fun(sysi, est_params)
        
        del_acts0 = np.zeros(Nacts)
        res = minimize(m.val_and_grad, 
                        jac=True, 
                        x0=del_acts0,
                        args=(E_ab, E_target, E_model_nom, reg_cond), 
                        method='L-BFGS-B',
                        tol=bfgs_tol,
                        options=bfgs_opts,
                        )

        del_acts = gain * res.x
        del_command[dm_mask] = del_acts
        total_command += del_command
        image_ni = xp.abs(forward_model(total_command[dm_mask], use_vortex=True, use_wfe=True))**2 / I_max_ref

        mean_ni = xp.mean(image_ni[control_mask])

        metric_images.append(copy.copy(image_ni))
        ef_estimates.append(copy.copy(E_ab))
        dm_commands.append(copy.copy(total_command))

        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=psf_pixelscale_lamD, lognorm3=True, vmin3=1e-9)

    return metric_images, ef_estimates, dm_commands


