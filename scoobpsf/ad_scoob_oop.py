import numpy as np
import astropy.units as u
from astropy.io import fits

import os
from pathlib import Path
import time
import copy

import poppy

from scipy.signal import windows
from scipy.optimize import minimize

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))
from . import utils
from . import props
from .math_module import xp, _scipy, ensure_np_array
from .imshows import imshow1, imshow2, imshow3

def make_dm_mask(Nact):
    y,x = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
    r = xp.sqrt(x**2 + y**2)
    dm_mask = r<(Nact/2 + 1/2)
    return dm_mask

def make_gaussian_inf_fun(act_spacing=300e-6*u.m, 
                          sampling=25, 
                          Nacts_per_inf=4, # number of influence functions across the grid
                          coupling=0.15,
                          ):

    ng = int(sampling*Nacts_per_inf)

    pxscl = act_spacing/(sampling*u.pix)
    ext = Nacts_per_inf * act_spacing

    xs = (xp.linspace(-ng/2,ng/2-1,ng)+1/2)*pxscl.value
    x,y = xp.meshgrid(xs,xs)
    r = xp.sqrt(x**2 + y**2)

    # d = act_spacing.value/1.25
    d = act_spacing.value/xp.sqrt(-xp.log(coupling))
    # print(d)

    inf = xp.exp(-(r/d)**2)
    rcoupled = d*xp.sqrt(-xp.log(coupling))

    return inf

def make_inf_matrix(inf_fun, inf_sampling, dm_mask):
    # Must make sure inf_fun has the dimensions of Nsurf X Nsurf
    Nsurf = inf_fun.shape[0]
    Nact = dm_mask.shape[0]
    Nacts = int(dm_mask.sum())

    inf_matrix = xp.zeros((Nsurf**2, Nacts))
    count = 0
    for i in range(Nact):
        for j in range(Nact):
            if dm_mask[i,j]:
                x_shift = (j-Nact/2 + 1/2)*inf_sampling
                y_shift = (i-Nact/2 + 1/2)*inf_sampling
                inf_matrix[:,count] = _scipy.ndimage.shift(inf_fun, (y_shift,x_shift)).ravel()
                count += 1
    
    return inf_matrix


def acts_to_command(acts, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact,Nact))
    command[dm_mask] = acts
    return command

class MODEL():
    def __init__(self):

        # initialize physical parameters
        self.wavelength_c = 650e-9*u.m
        self.wavelength = 650e-9*u.m
        self.pupil_diam = 9.4*u.mm
        self.dm_beam_diam = 9.4*u.mm
        self.lyot_stop_diam = 8.6*u.mm
        self.lyot_ratio = 8.6/9.4
        self.im_fl = 280*u.mm
        self.um_per_lamD = (self.wavelength_c*self.im_fl/(self.lyot_stop_diam)).to(u.um)
        self.psf_pixelscale = 3.76*u.um/u.pix
        self.psf_pixelscale_lamD = (self.psf_pixelscale / self.um_per_lamD).decompose().value
        self.Imax_ref = 1

        self.crad = 34/2 * 9.4/10.2 * 8.6/9.4

        # initialize sampling parameters and load masks
        self.npix = 1000
        self.nlyot = int(np.ceil(self.lyot_stop_diam/self.pupil_diam * self.npix))
        if self.nlyot%2==1:
            self.nlyot += 1
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)
        self.npsf = 200

        self.WFE = xp.ones((self.npix,self.npix), dtype=complex)
        
        pwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=1) # pupil wavefront
        self.APERTURE = poppy.CircularAperture(radius=self.pupil_diam/2).get_transmission(pwf)
        self.APMASK = self.APERTURE>0
        self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.pupil_diam/2).get_transmission(pwf)

        # initialize DM parameters
        self.Nact = 34
        self.Nacts = 952
        self.act_spacing = 0.3*u.mm
        self.inf_sampling = (self.npix/self.dm_beam_diam * self.act_spacing).value

        self.inf_fun = make_gaussian_inf_fun(act_spacing=self.act_spacing, sampling=self.inf_sampling, 
                                             Nacts_per_inf=self.Nact + 2, # number of influence functions across the grid
                                             coupling=0.15,)
        self.Nsurf = self.inf_fun.shape[0]
        self.dm_mask = make_dm_mask(self.Nact)
        self.inf_matrix = make_inf_matrix(self.inf_fun, self.inf_sampling, self.dm_mask)

        # Vortex model parameters
        self.oversample_vortex = 4.096
        self.N_vortex_lres = int(self.npix*self.oversample_vortex)
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(30/self.lres_sampling) + 1
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)
        # imshow2(xp.angle(vortex_lres), 1-lres_window, npix1=64, npix2=lres_win_size, pxscl2=lres_sampling)

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(30.029296875/self.hres_sampling))
        self.hres_win_size = int(30.029296875/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2)*self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        sing_mask = r>=0.15
        self.hres_window *= sing_mask
        # imshow2(xp.angle(vortex_hres), hres_window, npix1=64, npix2=hres_win_size, pxscl2=hres_sampling)

        self.iwa = 3
        self.owa = 12
        self.edge = 3
        self.dh_rotation = 0

        self.reg_cond = 1e-2
    
    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl

    def create_control_mask(self):
        x = (xp.linspace(-self.npsf/2, self.npsf/2-1, self.npsf) + 1/2)*self.psf_pixelscale_lamD
        x,y = xp.meshgrid(x,x)
        r = xp.hypot(x, y)
        control_mask = (r < self.owa) * (r > self.iwa)
        if self.edge is not None: control_mask *= (x > self.edge)

        self.control_mask = _scipy.ndimage.rotate(control_mask, self.dh_rotation, reshape=False, order=0)
        self.Nmask = int(control_mask.sum())

    def forward(self, actuators, use_vortex=True, use_wfe=False, return_pupil=False):
        dm_surf = self.inf_matrix.dot(xp.array(actuators)).reshape(self.Nsurf,self.Nsurf)
        dm_phasor = xp.exp(1j * 4*xp.pi/self.wavelength.to_value(u.m) * dm_surf)

        wf = utils.pad_or_crop(self.APERTURE, self.N).astype(xp.complex128)
        wf *= utils.pad_or_crop(dm_phasor, self.N)
        # imshow2(xp.abs(wf), xp.angle(wf), npix=npix)

        if use_wfe: 
            wf *= utils.pad_or_crop(self.WFE, self.N)
            # imshow2(xp.abs(wf), xp.angle(wf), npix=npix)

        E_pup = copy.copy(wf)

        if use_vortex:
            lres_wf = utils.pad_or_crop(wf, self.N_vortex_lres) # pad to the larger array for the low res propagation
            fp_wf_lres = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(lres_wf))) # to FPM
            fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
            pupil_wf_lres = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fp_wf_lres))) # to Lyot Pupil
            # pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, N)

            hres_wf = utils.pad_or_crop(wf, self.npix) # crop to the pupil diameter for the high res propagation
            fp_wf_hres = props.mft_forward(hres_wf, self.hres_sampling, self.N_vortex_hres)
            fp_wf_hres *= self.vortex_hres * self.hres_window # apply high res (windowed) FPM
            # pupil_wf_hres = props.mft_reverse(fp_wf_hres, hres_sampling, npix,)
            # pupil_wf_hres = utils.pad_or_crop(pupil_wf_hres, N)
            pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling*self.oversample_vortex, self.N_vortex_lres,)

            wf = (pupil_wf_lres + pupil_wf_hres)
            wf = utils.pad_or_crop(wf, self.N)
            # imshow2(xp.abs(wf), xp.angle(wf))

        wf *= utils.pad_or_crop(self.LYOT, self.N)
        # imshow2(xp.abs(wf), xp.angle(wf), npix=2*npix)

        wf = utils.pad_or_crop(wf, self.nlyot)
        fpwf = props.mft_forward(wf, self.psf_pixelscale_lamD, self.npsf) / xp.sqrt(self.Imax_ref)

        if return_pupil:
            return fpwf, E_pup
        else:
            return fpwf
    
# def val_and_grad(del_acts, m, E_ab, r_cond, E_target=0, E_model_nom=0,  verbose=False):
#     # Convert array arguments into GPU arrays if necessary
#     E_ab = xp.array(E_ab)
#     E_target = xp.array(E_target)
#     E_model_nom = xp.array(E_model_nom)
    
#     E_ab_l2norm = E_ab[m.control_mask].dot(E_ab[m.control_mask].conjugate()).real

#     # Compute E_dm using the forward DM model
#     # E_model_nom, E_pup = self.forward(actuators, use_vortex=True, use_wfe=False, return_pupil=True) # make sure to do the array indexing
#     # E_dm = self.forward(actuators+del_acts, use_vortex=True, use_wfe=False) # make sure to do the array indexing
#     # E_dm = E_dm - E_model_nom

#     # E_model_nom, E_pup = m.forward(actuators, use_vortex=True, use_wfe=True, return_pupil=True) # make sure to do the array indexing
#     # E_dm = m.forward(actuators+del_acts, use_vortex=True, use_wfe=True) # make sure to do the array indexing
#     # E_dm = E_dm - E_model_nom

#     E_dm = m.forward(del_acts, use_vortex=True, use_wfe=False) # make sure to do the array indexing
#     E_dm = E_dm - E_model_nom

#     _, E_pup = m.forward(np.zeros(m.Nacts), use_vortex=True, use_wfe=True, return_pupil=True)

#     # compute the cost function
#     delE = E_ab + E_dm - E_target
#     delE_vec = delE[m.control_mask] # make sure to do array indexing
#     J_delE = delE_vec.dot(delE_vec.conjugate()).real
#     # M_tik = r_cond * np.eye(m.Nacts, m.Nacts)
#     # c = M_tik.dot(del_acts) # I think I am doing something wrong with M_tik
#     # J_c = c.dot(c)
#     J_c = del_acts.dot(del_acts) * r_cond**2 / (m.wavelength.to_value(u.m))**2
#     J = (J_delE + J_c) / E_ab_l2norm
#     if verbose: print(J_delE, J_c, E_ab_l2norm, J)

#     # Compute the gradient with the adjoint model
#     delE_masked = m.control_mask * delE # still a 2D array
#     dJ_dE_dm = 2 * delE_masked / E_ab_l2norm
#     dJ_dE_ls = props.mft_reverse(dJ_dE_dm, m.psf_pixelscale_lamD, m.nlyot)
#     dJ_dE_ls = utils.pad_or_crop(dJ_dE_ls, m.npix)
#     dJ_dE_lp = m.LYOT * dJ_dE_ls
#     # imshow2(xp.abs(dJ_dE_lp), xp.angle(dJ_dE_lp))

#     # Now we have to split and back-propagate the gradient along the two branches 
#     # used to model the vortex. So one branch for the FFT vortex procedure and one 
#     # for the MFT vortex procedure. 
#     dJ_dE_lp_fft = utils.pad_or_crop(copy.copy(dJ_dE_lp), m.N_vortex_lres)
#     dJ_dE_fpm_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dE_lp_fft)))
#     dJ_dE_fp_fft = m.vortex_lres.conjugate() * (1 - m.lres_window) * dJ_dE_fpm_fft
#     dJ_dE_pup_fft = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(dJ_dE_fp_fft)))
#     dJ_dE_pup_fft = utils.pad_or_crop(dJ_dE_pup_fft, m.N)
#     # imshow2(xp.abs(dJ_dE_pup_fft), xp.angle(dJ_dE_pup_fft), npix=1*npix)

#     dJ_dE_lp_mft = utils.pad_or_crop(copy.copy(dJ_dE_lp), m.N_vortex_lres)
#     dJ_dE_fpm_mft = props.mft_forward(dJ_dE_lp_mft, m.hres_sampling*m.oversample_vortex, m.N_vortex_hres)
#     # dJ_dE_lp_mft = utils.pad_or_crop(copy.copy(dJ_dE_lp), nlyot)
#     # dJ_dE_fpm_mft = props.mft_forward(dJ_dE_lp_mft, hres_sampling, N_vortex_hres)
#     dJ_dE_fp_mft = m.vortex_hres.conjugate() * m.hres_window * dJ_dE_fpm_mft
#     # dJ_dE_pup_mft = props.mft_reverse(dJ_dE_fp_mft, hres_sampling, npix,)
#     dJ_dE_pup_mft = props.mft_reverse(dJ_dE_fp_mft, m.hres_sampling*m.oversample, m.N,)
#     # dJ_dE_pup_mft = utils.pad_or_crop(dJ_dE_pup_mft, N)
#     # imshow2(xp.abs(dJ_dE_pup_mft), xp.angle(dJ_dE_pup_mft), npix=1*npix)

#     dJ_dE_pup = dJ_dE_pup_fft + dJ_dE_pup_mft
#     # imshow2(xp.abs(dJ_dE_pup), xp.angle(dJ_dE_pup), npix=1*npix)

#     dJ_dS_dm = 4*xp.pi / m.wavelength.to_value(u.m) * xp.imag(E_pup.conjugate()/xp.sqrt(m.Imax_ref) * dJ_dE_pup)

#     # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
#     dJ_dS_dm = utils.pad_or_crop(dJ_dS_dm, m.Nsurf)
#     dJ_dA = m.inf_matrix.T.dot(dJ_dS_dm.flatten()) + xp.array( 2*del_acts * r_cond**2 / (m.wavelength.to_value(u.m))**2 )

#     return ensure_np_array(J), ensure_np_array(dJ_dA)


def val_and_grad(del_acts, m, actuators, E_ab, r_cond, verbose=False):
    # Convert array arguments into GPU arrays if necessary
    E_ab = xp.array(E_ab)
    
    E_ab_l2norm = E_ab[m.control_mask].dot(E_ab[m.control_mask].conjugate()).real

    # Compute E_dm using the forward DM model
    E_model_nom, E_pup = m.forward(actuators, use_vortex=True, use_wfe=True, return_pupil=True) # make sure to do the array indexing
    E_dm = m.forward(actuators+del_acts, use_vortex=True, use_wfe=True) # make sure to do the array indexing
    E_dm = E_dm - E_model_nom

    # compute the cost function
    delE = E_ab + E_dm
    delE_vec = delE[m.control_mask] # make sure to do array indexing
    J_delE = delE_vec.dot(delE_vec.conjugate()).real
    # M_tik = r_cond * np.eye(m.Nacts, m.Nacts)
    # c = M_tik.dot(del_acts) # I think I am doing something wrong with M_tik
    # J_c = c.dot(c)
    J_c = del_acts.dot(del_acts) * r_cond**2 / (m.wavelength.to_value(u.m))**2
    J = (J_delE + J_c) / E_ab_l2norm
    if verbose: print(J_delE, J_c, E_ab_l2norm, J)

    # Compute the gradient with the adjoint model
    delE_masked = m.control_mask * delE # still a 2D array
    dJ_dE_dm = 2 * delE_masked / E_ab_l2norm
    dJ_dE_ls = props.mft_reverse(dJ_dE_dm, m.psf_pixelscale_lamD, m.nlyot)
    dJ_dE_ls = utils.pad_or_crop(dJ_dE_ls, m.npix)
    dJ_dE_lp = m.LYOT * dJ_dE_ls
    # imshow2(xp.abs(dJ_dE_lp), xp.angle(dJ_dE_lp))

    # Now we have to split and back-propagate the gradient along the two branches 
    # used to model the vortex. So one branch for the FFT vortex procedure and one 
    # for the MFT vortex procedure. 
    dJ_dE_lp_fft = utils.pad_or_crop(copy.copy(dJ_dE_lp), m.N_vortex_lres)
    dJ_dE_fpm_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dE_lp_fft)))
    dJ_dE_fp_fft = m.vortex_lres.conjugate() * (1 - m.lres_window) * dJ_dE_fpm_fft
    dJ_dE_pup_fft = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(dJ_dE_fp_fft)))
    dJ_dE_pup_fft = utils.pad_or_crop(dJ_dE_pup_fft, m.N)
    # imshow2(xp.abs(dJ_dE_pup_fft), xp.angle(dJ_dE_pup_fft), npix=1*npix)

    dJ_dE_lp_mft = utils.pad_or_crop(copy.copy(dJ_dE_lp), m.N_vortex_lres)
    dJ_dE_fpm_mft = props.mft_forward(dJ_dE_lp_mft, m.hres_sampling*m.oversample_vortex, m.N_vortex_hres)
    # dJ_dE_lp_mft = utils.pad_or_crop(copy.copy(dJ_dE_lp), nlyot)
    # dJ_dE_fpm_mft = props.mft_forward(dJ_dE_lp_mft, hres_sampling, N_vortex_hres)
    dJ_dE_fp_mft = m.vortex_hres.conjugate() * m.hres_window * dJ_dE_fpm_mft
    # dJ_dE_pup_mft = props.mft_reverse(dJ_dE_fp_mft, hres_sampling, npix,)
    dJ_dE_pup_mft = props.mft_reverse(dJ_dE_fp_mft, m.hres_sampling*m.oversample, m.N,)
    # dJ_dE_pup_mft = utils.pad_or_crop(dJ_dE_pup_mft, N)
    # imshow2(xp.abs(dJ_dE_pup_mft), xp.angle(dJ_dE_pup_mft), npix=1*npix)

    dJ_dE_pup = dJ_dE_pup_fft + dJ_dE_pup_mft
    # imshow2(xp.abs(dJ_dE_pup), xp.angle(dJ_dE_pup), npix=1*npix)

    dJ_dS_dm = 4*xp.pi / m.wavelength.to_value(u.m) * xp.imag(E_pup.conjugate()/xp.sqrt(m.Imax_ref) * dJ_dE_pup)

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_dm = utils.pad_or_crop(dJ_dS_dm, m.Nsurf)
    dJ_dA = m.inf_matrix.T.dot(dJ_dS_dm.flatten()) + xp.array( 2*del_acts * r_cond**2 / (m.wavelength.to_value(u.m))**2 )

    return ensure_np_array(J), ensure_np_array(dJ_dA)


