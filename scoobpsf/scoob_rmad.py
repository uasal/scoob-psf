from .math_module import xp, _scipy, ensure_np_array
import scoobpsf.utils as utils
import scoobpsf.imshows as imshows
import scoobpsf.props as props
import scoobpsf.dm as dm

import numpy as np
import astropy.units as u
from astropy.io import fits
import os
from pathlib import Path
import time
import copy

import poppy
from scipy.signal import windows

def acts_to_command(acts, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact,Nact))
    command[dm_mask] = xp.array(acts)
    return command

class MODEL():
    def __init__(self, 
                 dm_beam_diam=9.2*u.mm,
                 lyot_pupil_diam=9.2*u.mm,
                 lyot_stop_diam=8.6*u.mm,
                 dm_shift=np.array([0, 0])*u.mm,
                 lyot_shift=np.array([0, 0])*u.mm,
                 ):

        # initialize physical parameters
        self.wavelength_c = 633e-9*u.m
        self.dm_beam_diam = dm_beam_diam
        self.lyot_pupil_diam = lyot_pupil_diam
        self.lyot_stop_diam = lyot_stop_diam
        self.lyot_ratio = (self.lyot_stop_diam / self.lyot_pupil_diam).decompose().value
        self.control_rad = 34/2 * self.dm_beam_diam.to_value(u.mm)/10.2 * self.lyot_ratio
        self.psf_pixelscale_lamDc = 0.307
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc
        self.npsf = 150

        self.wavelength = 633e-9*u.m

        self.Imax_ref = 1

        # initialize sampling parameters and load masks
        self.npix = 1000
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)

        self.dm_pxscl = self.dm_beam_diam.to_value(u.m)/self.npix
        self.lyot_pxscl = self.lyot_pupil_diam.to_value(u.m)/self.npix

        self.dm_shift = dm_shift
        self.lyot_shift = lyot_shift
        self.dm_shift_pix = self.dm_shift.to_value(u.m) / self.dm_pxscl
        self.lyot_shift_pix = self.lyot_shift.to_value(u.m) / self.lyot_pxscl

        pwf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=1) # pupil wavefront
        self.APERTURE = poppy.CircularAperture(radius=self.dm_beam_diam/2).get_transmission(pwf)
        self.APMASK = self.APERTURE>0
        self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.dm_beam_diam/2).get_transmission(pwf)
        self.LYOT = _scipy.ndimage.shift(self.LYOT, np.flip(self.lyot_shift_pix), order=1)
        self.WFE = xp.ones((self.npix,self.npix), dtype=complex)

        self.det_rotation = 0
        self.flip_dm = False
        self.reverse_lyot = False
        self.flip_lyot = False

        self.Nact = 34
        self.dm_shape = (self.Nact, self.Nact)
        self.act_spacing = 300e-6*u.m
        self.inf_sampling = self.act_spacing.to_value(u.m)/self.dm_pxscl
        self.inf_fun = dm.make_gaussian_inf_fun(act_spacing=self.act_spacing, sampling=self.inf_sampling, coupling=0.15, Nact=self.Nact+2)
        self.Nsurf = self.inf_fun.shape[0]

        y,x = (xp.indices((self.Nact, self.Nact)) - self.Nact//2 + 1/2)
        r = xp.sqrt(x**2 + y**2)
        self.dm_mask = r<(self.Nact/2 + 1/2)
        self.dm_mask[25,21] = False
        self.Nacts = int(self.dm_mask.sum())

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
        self.lres_win_size = int(30/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(30/self.hres_sampling))
        self.hres_win_size = int(30/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2)*self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        self.hres_dot_mask = r>=0.15

        self.use_vortex = True
        self.dm_command = xp.zeros((self.Nact, self.Nact))
    
    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc * (self.wavelength_c/wl).decompose().value

    def forward(self, actuators, use_vortex=True, return_ints=False, plot=False):
        dm_command = xp.zeros((self.Nact,self.Nact))
        dm_command[self.dm_mask] = xp.array(actuators)
        mft_command = self.Mx@dm_command@self.My
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fourier_surf,))).real
        DM_PHASOR = xp.exp(1j * 4*xp.pi/self.wavelength.to_value(u.m) * utils.pad_or_crop(dm_surf, self.N))
        DM_PHASOR = _scipy.ndimage.shift(DM_PHASOR, np.flip(self.dm_shift_pix), order=5)
        # if self.flip_dm: DM_PHASOR = xp.rot90(xp.rot90(DM_PHASOR))

        # Initialize the wavefront
        E_EP = utils.pad_or_crop(self.APERTURE.astype(xp.complex128), self.N) * utils.pad_or_crop(self.WFE, self.N) / xp.sqrt(self.Imax_ref)
        if plot: imshows.imshow2(xp.abs(E_EP), xp.angle(E_EP), 'EP WF', npix=1.5*self.npix, cmap2='twilight')

        E_DM = E_EP * utils.pad_or_crop(DM_PHASOR, self.N)
        if plot: imshows.imshow2(xp.abs(E_DM), xp.angle(E_DM), 'After DM WF', npix=1.5*self.npix, cmap2='twilight')

        if use_vortex:
            lres_wf = utils.pad_or_crop(E_DM, self.N_vortex_lres) # pad to the larger array for the low res propagation
            fp_wf_lres = props.fft(lres_wf)
            fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res FPM and inverse Tukey window
            pupil_wf_lres = props.ifft(fp_wf_lres)
            pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N)
            if plot: imshows.imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 'FFT Lyot WF', npix=1.5*self.npix, cmap2='twilight')

            fp_wf_hres = props.mft_forward(E_DM, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
            fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res FPM, window, and dot mask
            pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+')
            if plot: imshows.imshow2(xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 'MFT Lyot WF', npix=1.5*self.npix, cmap2='twilight')

            E_LP = (pupil_wf_lres + pupil_wf_hres)
            if plot: imshows.imshow2(xp.abs(E_LP), xp.angle(E_LP), 'Total Lyot WF', npix=1.5*self.npix, cmap2='twilight')
        else:
            E_LP = E_DM

        if self.reverse_lyot: E_LP = xp.rot90(xp.rot90(E_LP))
        if self.flip_lyot: E_LP = xp.fliplr(E_LP)
        # E_LP = _scipy.ndimage.shift(E_LP, np.flip(self.lyot_shift_pix), order=5)

        E_LS = utils.pad_or_crop(self.LYOT, self.N) * E_LP
        if plot: imshows.imshow2(xp.abs(E_LS), xp.angle(E_LS), 'After Lyot Stop WF', npix=1.5*self.npix, cmap2='twilight')
        
        fpwf = props.mft_forward(E_LS, self.npix * self.lyot_ratio, self.npsf, self.psf_pixelscale_lamD)
        fpwf = _scipy.ndimage.rotate(fpwf, self.det_rotation, reshape=False, order=5)
        if plot: imshows.imshow2(xp.abs(fpwf)**2, xp.angle(fpwf), 'At SCICAM WF', lognorm1=True, cmap2='twilight')

        if return_ints:
            return fpwf, E_EP, DM_PHASOR, E_LP, E_LS
        else:
            return fpwf
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)
        
    def zero_dm(self):
        self.dm_command = xp.zeros((self.Nact,self.Nact))

    def add_dm(self, del_dm):
        self.dm_command += del_dm
    
    def set_dm(self, dm_command):
        self.dm_command = dm_command

    def get_dm(self,):
        return copy.copy(self.dm_command)
        
    def calc_wf(self):
        actuators = self.dm_command[self.dm_mask]
        fpwf = self.forward(actuators, use_vortex=self.use_vortex,)
        return fpwf
        
    def snap(self):
        im = xp.abs( self.calc_wf() )**2
        return im

def val_and_grad(del_acts, M, actuators, E_ab, r_cond, control_mask, verbose=False, plot=False):
    # Convert array arguments into correct types
    actuators = ensure_np_array(actuators)
    del_acts_waves = del_acts/M.wavelength_c.to_value(u.m)
    E_ab = xp.array(E_ab)
    
    E_ab_l2norm = E_ab[control_mask].dot(E_ab[control_mask].conjugate()).real

    # Compute E_dm using the forward DM model
    E_FP_NOM, E_EP, DM_PHASOR, _, _ = M.forward(actuators, use_vortex=True, return_ints=True) # make sure to do the array indexing
    E_FP_w_DM = M.forward(actuators + del_acts, use_vortex=True) # make sure to do the array indexing
    E_DM = E_FP_w_DM - E_FP_NOM

    # compute the cost function
    delE = E_ab + E_DM
    delE_vec = delE[control_mask] # make sure to do array indexing
    J_delE = delE_vec.dot(delE_vec.conjugate()).real
    J_c = r_cond * del_acts_waves.dot(del_acts_waves)
    J = (J_delE + J_c) / E_ab_l2norm
    if verbose: 
        print(f'\tCost-function J_delE: {J_delE:.3f}')
        print(f'\tCost-function J_c: {J_c:.3f}')
        print(f'\tCost-function normalization factor: {E_ab_l2norm:.3f}')
        print(f'\tTotal cost-function value: {J:.3f}\n')

    # Compute the gradient with the adjoint model
    delE_masked = control_mask * delE # still a 2D array
    delE_masked = _scipy.ndimage.rotate(delE_masked, -M.det_rotation, reshape=False, order=5)
    dJ_dE_DM = 2 * delE_masked / E_ab_l2norm

    dJ_dE_LS = props.mft_reverse(dJ_dE_DM, M.psf_pixelscale_lamD, M.npix * M.lyot_ratio, M.N, convention='+')
    if plot: imshows.imshow2(xp.abs(dJ_dE_LS), xp.angle(dJ_dE_LS), 'RMAD Lyot Stop', npix=1.5*M.npix, cmap2='twilight')

    dJ_dE_LP = dJ_dE_LS * utils.pad_or_crop(M.LYOT, M.N)
    if M.flip_lyot: dJ_dE_LP = xp.fliplr(dJ_dE_LP)
    if M.reverse_lyot: dJ_dE_LP = xp.rot90(xp.rot90(dJ_dE_LP))
    if plot: imshows.imshow2(xp.abs(dJ_dE_LP), xp.angle(dJ_dE_LP), 'RMAD Lyot Pupil', npix=1.5*M.npix, cmap2='twilight')

    # Now we have to split and back-propagate the gradient along the two branches used to model the vortex.
    # So one branch for the FFT vortex procedure and one for the MFT vortex procedure. 
    dJ_dE_LP_fft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N_vortex_lres)
    dJ_dE_FPM_fft = props.fft(dJ_dE_LP_fft)
    dJ_dE_FP_fft = M.vortex_lres.conj() * (1 - M.lres_window) * dJ_dE_FPM_fft
    dJ_dE_PUP_fft = props.ifft(dJ_dE_FP_fft)
    dJ_dE_PUP_fft = utils.pad_or_crop(dJ_dE_PUP_fft, M.N)
    if plot: imshows.imshow2(xp.abs(dJ_dE_PUP_fft), xp.angle(dJ_dE_PUP_fft), 'RMAD FFT Pupil', npix=1.5*M.npix, cmap2='twilight')

    dJ_dE_LP_mft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N)
    dJ_dE_FPM_mft = props.mft_forward(dJ_dE_LP_mft,  M.npix, M.N_vortex_hres, M.hres_sampling, convention='-')
    dJ_dE_FP_mft = M.vortex_hres.conj() * M.hres_window * M.hres_dot_mask * dJ_dE_FPM_mft
    dJ_dE_PUP_mft = props.mft_reverse(dJ_dE_FP_mft, M.hres_sampling, M.npix, M.N, convention='+')
    if plot: imshows.imshow2(xp.abs(dJ_dE_PUP_mft), xp.angle(dJ_dE_PUP_mft), 'RMAD MFT Pupil', npix=1.5*M.npix, cmap2='twilight')

    dJ_dE_PUP = dJ_dE_PUP_fft + dJ_dE_PUP_mft
    if plot: imshows.imshow2(xp.abs(dJ_dE_PUP), xp.angle(dJ_dE_PUP), 'RMAD Total Pupil', npix=1.5*M.npix, cmap2='twilight')

    dJ_dS_DM = 4*xp.pi / M.wavelength.to_value(u.m) * xp.imag(dJ_dE_PUP * E_EP.conj() * DM_PHASOR.conj())
    # if M.flip_dm: dJ_dS_DM = xp.rot90(xp.rot90(dJ_dS_DM))
    if plot: imshows.imshow2(xp.real(dJ_dS_DM), xp.imag(dJ_dS_DM), 'RMAD DM Surface', npix=1.5*M.npix)

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_DM = utils.pad_or_crop(dJ_dS_DM, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM.real)))
    x1_bar = M.inf_fun_fft.conjugate() * x2_bar
    dJ_dA = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: imshows.imshow2(dJ_dA.real, dJ_dA.imag, 'RMAD DM Actuators')
    dJ_dA = dJ_dA[M.dm_mask].real + xp.array( r_cond * 2*del_acts_waves )

    return ensure_np_array(J), ensure_np_array(dJ_dA)



