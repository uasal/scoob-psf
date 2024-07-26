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

def sim_pwp(m, current_acts, 
            probes, probe_amp, 
            reg_cond=1e-3, 
            plot=False,
            plot_est=False):
    
    Nmask = int(m.control_mask.sum())
    Nprobes = probes.shape[0]

    Ip = []
    In = []
    for i in range(Nprobes):
        for s in [-1, 1]:
            coro_im = m.snap(current_acts + s*probe_amp*probes[i][m.dm_mask], use_vortex=True, use_wfe=True)

            if s==-1: 
                In.append(coro_im)
            else: 
                Ip.append(coro_im)
            
        if plot:
            imshow3(Ip[i], In[i], Ip[i]-In[i], lognorm1=True, lognorm2=True, pxscl=m.psf_pixelscale_lamD)
            
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(Nprobes):
        if i==0: 
            E_nom = m.forward(current_acts, use_vortex=True, use_wfe=True)
        E_with_probe = m.forward(current_acts + probe_amp*probes[i][m.dm_mask], use_vortex=True, use_wfe=True)
        E_probe = E_with_probe - E_nom

        if plot:
            imshow2(xp.abs(E_probe), xp.angle(E_probe),
                    f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$')
            
        E_probes[i, ::2] = E_probe[m.control_mask].real
        E_probes[i, 1::2] = E_probe[m.control_mask].imag

        # I_diff[i:(i+1), :] = (Ip[i] - In[i])[control_mask]
        I_diff[i, :] = (Ip[i] - In[i])[m.control_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        M = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Minv = xp.linalg.pinv(M.T@M, reg_cond)@M.T
    
        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((m.npsf,m.npsf), dtype=xp.complex128)
    E_est_2d[m.control_mask] = E_est
    
    if plot or plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        imshow2(I_est, P_est, 
                'Estimated Intensity', 'Estimated Phase',
                lognorm1=True, vmin1=xp.max(I_est)/1e4, 
                cmap2='twilight',
                pxscl=m.psf_pixelscale_lamD)
    return E_est_2d

def run_pwp(sysi, m, current_acts, 
            control_mask, 
            probes, probe_amp, 
            reg_cond=1e-3, 
            plot=False,
            plot_est=False):
    
    Nmask = int(control_mask.sum())
    Nprobes = probes.shape[0]

    Ip = []
    In = []
    for i in range(Nprobes):
        for s in [-1, 1]:
            sysi.add_dm(s*probe_amp*probes[i])
            coro_im = sysi.snap()
            sysi.add_dm(-s*probe_amp*probes[i]) # remove probe from DM

            if s==-1: 
                In.append(coro_im)
            else: 
                Ip.append(coro_im)
            
        if plot:
            imshow3(Ip[i], In[i], Ip[i]-In[i], lognorm1=True, lognorm2=True, pxscl=sysi.psf_pixelscale_lamD)
            
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(Nprobes):
        if i==0: 
            E_nom = m.forward(current_acts, use_vortex=True, use_wfe=True)
        E_with_probe = m.forward(xp.array(current_acts) + xp.array(probe_amp*probes[i])[m.dm_mask], use_vortex=True, use_wfe=True)
        E_probe = E_with_probe - E_nom

        if plot:
            imshow2(xp.abs(E_probe), xp.angle(E_probe),
                    f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$')
            
        E_probes[i, ::2] = E_probe[control_mask].real
        E_probes[i, 1::2] = E_probe[control_mask].imag

        # I_diff[i:(i+1), :] = (Ip[i] - In[i])[control_mask]
        I_diff[i, :] = (Ip[i] - In[i])[control_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        M = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Minv = xp.linalg.pinv(M.T@M, reg_cond)@M.T
    
        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((sysi.npsf,sysi.npsf), dtype=xp.complex128)
    E_est_2d[control_mask] = E_est

    if plot or plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        imshow2(I_est, P_est, 
                'Estimated Intensity', 'Estimated Phase',
                lognorm1=True, vmin1=xp.max(I_est)/1e4, 
                cmap2='twilight',
                pxscl=m.psf_pixelscale_lamD)
    return E_est_2d


def sim(m, val_and_grad,
        est_fun=None, est_params=None,
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
        if est_fun is None: 
            E_ab = m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True,)
        else: 
            E_ab = est_fun(m, total_command[m.dm_mask], **est_params)

        res = minimize(val_and_grad, 
                       jac=True, 
                       x0=del_acts0,
                       args=(m, ensure_np_array(total_command[m.dm_mask]), E_ab, reg_cond), 
                       method='L-BFGS-B',
                       tol=bfgs_tol,
                       options=bfgs_opts,
                       )

        del_acts = gain * res.x
        del_command[m.dm_mask] = del_acts
        total_command += del_command
        # image_ni = xp.abs(m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True))**2
        image_ni = m.snap(total_command[m.dm_mask])
        mean_ni = xp.mean(image_ni[m.control_mask])

        all_ims.append(copy.copy(image_ni))
        all_efs.append(copy.copy(E_ab))
        all_commands.append(copy.copy(total_command))

        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=m.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-10)

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
                pxscl3=m.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-10)

    return all_ims, all_efs, all_commands


