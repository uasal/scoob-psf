from .math_module import xp, _scipy, ensure_np_array
import scoobpsf.utils as utils
import scoobpsf.imshows as imshows
import scoobpsf.props as props
import scoobpsf.dm as dm

import numpy as np
import astropy.units as u
import copy
import poppy

from scipy.signal import windows
from scipy.optimize import minimize

def acts_to_command(acts, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact,Nact))
    command[dm_mask] = xp.array(acts)
    return command

class MODEL():
    def __init__(self):

        # initialize physical parameters
        self.wavelength_c = 633e-9*u.m
        self.dm_beam_diam = 9.2*u.mm
        self.lyot_pupil_diam = 9.2*u.mm
        self.lyot_stop_diam = 8.7*u.mm
        self.lyot_ratio = (self.lyot_stop_diam / self.lyot_pupil_diam).decompose().value
        self.control_rad = 34/2 * 9.2/10.2 * 8.7/9.2
        self.psf_pixelscale_lamDc = 0.307
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc
        self.npsf = 150

        self.wavelength = 633e-9*u.m

        self.Imax_ref = 1

        # initialize sampling parameters and load masks
        self.npix = 1000
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)

        pwf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=1) # pupil wavefront
        self.APERTURE = poppy.CircularAperture(radius=self.dm_beam_diam/2).get_transmission(pwf)
        self.APMASK = self.APERTURE>0
        self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.dm_beam_diam/2).get_transmission(pwf)
        self.WFE = xp.ones((self.npix,self.npix), dtype=complex)

        self.Nact = 34
        self.act_spacing = 300e-6*u.m
        self.dm_pxscl = self.dm_beam_diam.to_value(u.m)/self.npix
        self.inf_sampling = self.act_spacing.to_value(u.m)/self.dm_pxscl
        self.inf_fun = dm.make_gaussian_inf_fun(act_spacing=self.act_spacing, sampling=self.inf_sampling, coupling=0.15, Nact=self.Nact+2)
        self.Nsurf = self.inf_fun.shape[0]

        y,x = (xp.indices((self.Nact, self.Nact)) - self.Nact//2 + 1/2)
        r = xp.sqrt(x**2 + y**2)
        self.dm_mask = r<(self.Nact/2 + 1/2)
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

        self.det_rotation = 0

        self.flip_dm_lr = False
        self.flip_dm_ud = False
        self.flip_lyot_lr = False
        self.flip_lyot_ud = False

        self.use_vortex = True
        self.use_wfe = True
        self.dm_command = xp.zeros((self.Nact, self.Nact))
    
    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc * (self.wavelength_c/wl).decompose().value

    def forward(self, actuators, use_vortex=True, use_wfe=True, return_pupil=False, plot=False):
        dm_command = xp.zeros((self.Nact,self.Nact))
        dm_command[self.dm_mask] = xp.array(actuators)
        mft_command = self.Mx@dm_command@self.My
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(fourier_surf,))).real
        dm_phasor = xp.exp(1j * 4*xp.pi/self.wavelength.to_value(u.m) * dm_surf)

        wf = utils.pad_or_crop(self.APERTURE.astype(xp.complex128), self.N)
        wf *= utils.pad_or_crop(dm_phasor, self.N)
        if self.flip_dm_lr: wf = xp.fliplr(wf)
        if self.flip_dm_ud: wf = xp.flipud(wf)
        if plot: imshows.imshow2(xp.abs(wf), xp.angle(wf), npix=1.5*self.npix)

        if use_wfe: 
            wf *= utils.pad_or_crop(self.WFE, self.N)
            if plot: imshows.imshow2(xp.abs(wf), xp.angle(wf), npix=1.5*self.npix)

        if return_pupil: 
            E_pup = copy.copy(wf)

        if use_vortex:
            lres_wf = utils.pad_or_crop(wf, self.N_vortex_lres) # pad to the larger array for the low res propagation
            fp_wf_lres = props.fft(lres_wf)
            fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res FPM and inverse Tukey window
            pupil_wf_lres = props.ifft(fp_wf_lres)
            pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N)
            if plot: imshows.imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), npix=1.5*self.npix)

            fp_wf_hres = props.mft_forward(wf, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
            fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res FPM, window, and dot mask
            pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+')
            if plot: imshows.imshow2(xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), npix=1.5*self.npix)

            wf = (pupil_wf_lres + pupil_wf_hres)
            if plot: imshows.imshow2(xp.abs(wf), xp.angle(wf), npix=1.5*self.npix)

        if self.flip_lyot_lr: wf = xp.fliplr(wf)
        if self.flip_lyot_ud: wf = xp.flipud(wf)
        wf *= utils.pad_or_crop(self.LYOT, self.N)
        if plot: imshows.imshow2(xp.abs(wf), xp.angle(wf), npix=self.npix)

        fpwf = props.mft_forward(wf, self.npix * self.lyot_ratio, self.npsf, self.psf_pixelscale_lamD) / xp.sqrt(self.Imax_ref)

        fpwf = _scipy.ndimage.rotate(fpwf, self.det_rotation, reshape=False, order=5)
        if plot: imshows.imshow2(xp.abs(fpwf)**2, xp.angle(fpwf), lognorm1=True)

        if return_pupil:
            return fpwf, E_pup
        else:
            return fpwf
        
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
        fpwf = self.forward(actuators, use_vortex=self.use_vortex, use_wfe=self.use_wfe,)
        return fpwf
        
    def snap(self):
        actuators = self.dm_command[self.dm_mask]
        im = xp.abs(self.forward(actuators, use_vortex=self.use_vortex, use_wfe=self.use_wfe,))**2
        return im
    


def val_and_grad(del_acts, M, actuators, E_ab, r_cond, control_mask, verbose=False, plot=False):
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
    if verbose: 
        print(f'\tCost-function J_delE: {J_delE:.3f}') 
        print(f'\tCost-function J_c: {J_c:.3f}') 
        print(f'\tCost-function normalization factor: {E_ab_l2norm:.3f}')
        print(f'\tTotal cost-function value: {J:.3f}\n')

    # Compute the gradient with the adjoint model
    delE_masked = control_mask * delE # still a 2D array
    delE_masked = _scipy.ndimage.rotate(delE_masked, -M.det_rotation, reshape=False, order=5)
    dJ_dE_dm = 2 * delE_masked / E_ab_l2norm
    dJ_dE_ls = props.mft_reverse(dJ_dE_dm, M.psf_pixelscale_lamD, M.npix * M.lyot_ratio, M.N, convention='+')
    if plot: imshows.imshow2(xp.abs(dJ_dE_ls), xp.angle(dJ_dE_ls), npix=1*M.npix)
    dJ_dE_lp = utils.pad_or_crop(M.LYOT, M.N) * dJ_dE_ls
    if M.flip_lyot_ud: dJ_dE_lp = xp.flipud(dJ_dE_lp)
    if M.flip_lyot_lr: dJ_dE_lp = xp.fliplr(dJ_dE_lp)
    if plot: imshows.imshow2(xp.abs(dJ_dE_lp), xp.angle(dJ_dE_lp), npix=1*M.npix)

    # Now we have to split and back-propagate the gradient along the two branches 
    # used to model the vortex. So one branch for the FFT vortex procedure and one 
    # for the MFT vortex procedure. 
    dJ_dE_lp_fft = utils.pad_or_crop(copy.copy(dJ_dE_lp), M.N_vortex_lres)
    dJ_dE_fpm_fft = props.fft(dJ_dE_lp_fft)
    dJ_dE_fp_fft = M.vortex_lres.conjugate() * (1 - M.lres_window) * dJ_dE_fpm_fft
    dJ_dE_pup_fft = props.ifft(dJ_dE_fp_fft)
    dJ_dE_pup_fft = utils.pad_or_crop(dJ_dE_pup_fft, M.N)
    if plot: imshows.imshow2(xp.abs(dJ_dE_pup_fft), xp.angle(dJ_dE_pup_fft), npix=1.5*M.npix)

    dJ_dE_lp_mft = utils.pad_or_crop(copy.copy(dJ_dE_lp), M.N)
    dJ_dE_fpm_mft = props.mft_forward(dJ_dE_lp_mft,  M.npix, M.N_vortex_hres, M.hres_sampling, convention='-')
    dJ_dE_fp_mft = M.vortex_hres.conjugate() * M.hres_window * M.hres_dot_mask * dJ_dE_fpm_mft
    dJ_dE_pup_mft = props.mft_reverse(dJ_dE_fp_mft, M.hres_sampling, M.npix, M.N, convention='+')
    if plot: imshows.imshow2(xp.abs(dJ_dE_pup_mft), xp.angle(dJ_dE_pup_mft), npix=1.5*M.npix)

    dJ_dE_pup = dJ_dE_pup_fft + dJ_dE_pup_mft
    if plot: imshows.imshow2(xp.abs(dJ_dE_pup), xp.angle(dJ_dE_pup), npix=1*M.npix)
    
    if M.flip_dm_ud: dJ_dE_pup = xp.flipud(dJ_dE_pup)
    if M.flip_dm_lr: dJ_dE_pup = xp.fliplr(dJ_dE_pup)
    dJ_dS_dm = 4*xp.pi / M.wavelength.to_value(u.m) * xp.imag(E_pup.conjugate()/xp.sqrt(M.Imax_ref) * dJ_dE_pup)

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_dm = utils.pad_or_crop(dJ_dS_dm, M.Nsurf)
    # dJ_dA = m.inf_matrix.T.dot(dJ_dS_dm.flatten()) + xp.array( 2*del_acts * r_cond**2 / (m.wavelength.to_value(u.m))**2 )
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_dm.real)))
    x1_bar = M.inf_fun_fft.conjugate() * x2_bar
    dJ_dA = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: imshows.imshow2(dJ_dA.real, dJ_dA.imag)
    dJ_dA = dJ_dA[M.dm_mask].real + xp.array( 2*del_acts * r_cond / (M.wavelength_c.to_value(u.m))**2 )

    return ensure_np_array(J), ensure_np_array(dJ_dA)


