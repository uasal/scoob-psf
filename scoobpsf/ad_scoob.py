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
    command[dm_mask] = xp.array(acts)
    return command

class MODEL():
    def __init__(self,
                 use_dead_act=True,
                 ):

        # initialize physical parameters
        self.wavelength_c = 633e-9*u.m
        self.pupil_diam = 9.1*u.mm
        self.dm_beam_diam = 9.1*u.mm
        self.lyot_stop_diam = 8.6*u.mm
        self.lyot_ratio = 8.6/9.1
        self.crad = 34/2 * 9.1/10.2 * 8.6/9.1
        self.psf_pixelscale_lamDc = 0.307
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc
        self.npsf = 150

        self.wavelength = 633e-9*u.m

        self.Imax_ref = 1

        # initialize sampling parameters and load masks
        self.npix = 1000
        self.nlyot = int(np.ceil(self.lyot_stop_diam/self.pupil_diam * self.npix))
        if self.nlyot%2==1:
            self.nlyot += 1
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)

        self.WFE = xp.ones((self.npix,self.npix), dtype=complex)
        
        pwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=1) # pupil wavefront
        self.APERTURE = poppy.CircularAperture(radius=self.pupil_diam/2).get_transmission(pwf)
        self.APMASK = self.APERTURE>0
        self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.pupil_diam/2).get_transmission(pwf)

        # initialize DM parameters
        self.Nact = 34
        self.act_spacing = 0.3*u.mm
        self.inf_sampling = (self.npix/self.dm_beam_diam * self.act_spacing).value

        self.inf_fun = make_gaussian_inf_fun(act_spacing=self.act_spacing, sampling=self.inf_sampling, 
                                             Nacts_per_inf=self.Nact + 2, # number of influence functions across the grid
                                             coupling=0.15,)
        self.Nsurf = self.inf_fun.shape[0]

        self.dm_mask = make_dm_mask(self.Nact)
        if use_dead_act:
            self.dead_act = [25, 21]
            self.dm_mask[self.dead_act[0], self.dead_act[1]] = 0
        self.Nacts = int(self.dm_mask.sum())

        # self.inf_matrix = make_inf_matrix(self.inf_fun, self.inf_sampling, self.dm_mask)

        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))
        # DM command coordinates
        xc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)
        yc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        # Influence function frequncy sampling
        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))

        # forward DM model MFT matrices
        self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,xc))
        self.My = xp.exp(-1j*2*np.pi*xp.outer(yc,fy))

        self.Mx_back = xp.exp(1j*2*np.pi*xp.outer(xc,fx))
        self.My_back = xp.exp(1j*2*np.pi*xp.outer(fy,yc))

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

        self.det_rotation = 0

        self.reg_cond = 1e-2
    
    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc * (self.wavelength_c/wl).decompose().value

    def forward(self, actuators, use_vortex=True, use_wfe=True, return_pupil=False, plot=False):
        # dm_surf = self.inf_matrix.dot(xp.array(actuators)).reshape(self.Nsurf,self.Nsurf)
        dm_command = xp.zeros((self.Nact,self.Nact))
        dm_command[self.dm_mask] = xp.array(actuators)
        mft_command = self.Mx@dm_command@self.My
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(fourier_surf,))).real
        dm_phasor = xp.exp(1j * 4*xp.pi/self.wavelength.to_value(u.m) * dm_surf)

        wf = utils.pad_or_crop(self.APERTURE, self.N).astype(xp.complex128)
        wf *= utils.pad_or_crop(dm_phasor, self.N)
        if plot: imshow2(xp.abs(wf), xp.angle(wf), npix=self.npix)

        if use_wfe: 
            wf *= utils.pad_or_crop(self.WFE, self.N)
            if plot: imshow2(xp.abs(wf), xp.angle(wf), npix=self.npix)

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

        wf = xp.rot90(xp.rot90(wf)) # rotate array by 180 since we are going through focus
        wf = xp.fliplr(wf)
        wf *= utils.pad_or_crop(self.LYOT, self.N)
        if plot: imshow2(xp.abs(wf), xp.angle(wf), npix=self.npix)

        wf = utils.pad_or_crop(wf, self.nlyot)
        fpwf = props.mft_forward(wf, self.psf_pixelscale_lamD, self.npsf) / xp.sqrt(self.Imax_ref)

        fpwf = _scipy.ndimage.rotate(fpwf, self.det_rotation, reshape=False, order=5)
        if plot: imshow2(xp.abs(fpwf)**2, xp.angle(fpwf), lognorm1=True)

        if return_pupil:
            return fpwf, E_pup
        else:
            return fpwf
        
    def snap(self, actuators, use_vortex=True, use_wfe=True,):
        return xp.abs(self.forward(actuators, use_vortex=use_vortex, use_wfe=use_wfe,))**2

def val_and_grad(del_acts, M, actuators, E_ab, r_cond, control_mask, verbose=False):
    # Convert array arguments into correct types
    actuators = ensure_np_array(actuators)
    E_ab = xp.array(E_ab)
    
    E_ab_l2norm = E_ab[control_mask].dot(E_ab[control_mask].conjugate()).real

    # Compute E_dm using the forward DM model
    E_model_nom, E_pup = M.forward(actuators, use_vortex=True, use_wfe=True, return_pupil=True) # make sure to do the array indexing
    E_dm = M.forward(actuators+del_acts, use_vortex=True, use_wfe=True) # make sure to do the array indexing
    E_dm = E_dm - E_model_nom

    # compute the cost function
    delE = E_ab + E_dm
    delE_vec = delE[control_mask] # make sure to do array indexing
    J_delE = delE_vec.dot(delE_vec.conjugate()).real
    J_c = del_acts.dot(del_acts) * r_cond / (M.wavelength_c.to_value(u.m))**2
    J = (J_delE + J_c) / E_ab_l2norm
    if verbose: print(J_delE, J_c, E_ab_l2norm, J)

    # Compute the gradient with the adjoint model
    delE_masked = control_mask * delE # still a 2D array
    delE_masked = _scipy.ndimage.rotate(delE_masked, -M.det_rotation, reshape=False, order=5)
    dJ_dE_dm = 2 * delE_masked / E_ab_l2norm
    dJ_dE_ls = props.mft_reverse(dJ_dE_dm, M.psf_pixelscale_lamD, M.nlyot)
    dJ_dE_ls = utils.pad_or_crop(dJ_dE_ls, M.npix)
    dJ_dE_lp = M.LYOT * dJ_dE_ls
    dJ_dE_lp = xp.fliplr(dJ_dE_lp) # account for the parity flip from reflection
    dJ_dE_lp = xp.rot90(xp.rot90(dJ_dE_lp)) # account for the parity flip going through focus
    # imshow2(xp.abs(dJ_dE_lp), xp.angle(dJ_dE_lp))

    # Now we have to split and back-propagate the gradient along the two branches 
    # used to model the vortex. So one branch for the FFT vortex procedure and one 
    # for the MFT vortex procedure. 
    dJ_dE_lp_fft = utils.pad_or_crop(copy.copy(dJ_dE_lp), M.N_vortex_lres)
    dJ_dE_fpm_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dE_lp_fft)))
    dJ_dE_fp_fft = M.vortex_lres.conjugate() * (1 - M.lres_window) * dJ_dE_fpm_fft
    dJ_dE_pup_fft = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(dJ_dE_fp_fft)))
    dJ_dE_pup_fft = utils.pad_or_crop(dJ_dE_pup_fft, M.N)
    # imshow2(xp.abs(dJ_dE_pup_fft), xp.angle(dJ_dE_pup_fft), npix=1*npix)

    dJ_dE_lp_mft = utils.pad_or_crop(copy.copy(dJ_dE_lp), M.N_vortex_lres)
    dJ_dE_fpm_mft = props.mft_forward(dJ_dE_lp_mft, M.hres_sampling*M.oversample_vortex, M.N_vortex_hres)
    # dJ_dE_lp_mft = utils.pad_or_crop(copy.copy(dJ_dE_lp), nlyot)
    # dJ_dE_fpm_mft = props.mft_forward(dJ_dE_lp_mft, hres_sampling, N_vortex_hres)
    dJ_dE_fp_mft = M.vortex_hres.conjugate() * M.hres_window * dJ_dE_fpm_mft
    # dJ_dE_pup_mft = props.mft_reverse(dJ_dE_fp_mft, hres_sampling, npix,)
    dJ_dE_pup_mft = props.mft_reverse(dJ_dE_fp_mft, M.hres_sampling*M.oversample, M.N,)
    # dJ_dE_pup_mft = utils.pad_or_crop(dJ_dE_pup_mft, N)
    # imshow2(xp.abs(dJ_dE_pup_mft), xp.angle(dJ_dE_pup_mft), npix=1*npix)

    dJ_dE_pup = dJ_dE_pup_fft + dJ_dE_pup_mft
    # imshow2(xp.abs(dJ_dE_pup), xp.angle(dJ_dE_pup), npix=1*npix)

    dJ_dS_dm = 4*xp.pi / M.wavelength.to_value(u.m) * xp.imag(E_pup.conjugate()/xp.sqrt(M.Imax_ref) * dJ_dE_pup)

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_dm = utils.pad_or_crop(dJ_dS_dm, M.Nsurf)
    # dJ_dA = m.inf_matrix.T.dot(dJ_dS_dm.flatten()) + xp.array( 2*del_acts * r_cond**2 / (m.wavelength.to_value(u.m))**2 )
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_dm.real)))
    x1_bar = M.inf_fun_fft.conjugate() * x2_bar
    dJ_dA = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    dJ_dA = dJ_dA[M.dm_mask].real + xp.array( 2*del_acts * r_cond / (M.wavelength_c.to_value(u.m))**2 )

    return ensure_np_array(J), ensure_np_array(dJ_dA)


