import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import ray
import copy

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

from .import custom_dm
from .math_module import xp,_scipy, ensure_np_array
from . import imshows

def make_vortex_phase_mask(focal_grid_polar, charge=6, 
                           singularity=None, focal_length=500*u.mm, pupil_diam=9.7*u.mm, wavelength=632.8*u.nm):
    
    r = focal_grid_polar[0]
    th = focal_grid_polar[1]
    
    phasor = xp.exp(1j*charge*th)
    
    if singularity is not None:
#         sing*D/(focal_length*lam)
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose()
        phasor *= mask
    
    return phasor

import poppy
def generate_wfe(diam, distance=100*u.mm, 
                 opd_index=2.5, amp_index=2, 
                 opd_seed=1234, amp_seed=12345,
                 opd_rms=10*u.nm, amp_rms=0.05*u.nm,
                 npix=256, oversample=4, 
                 wavelength=500*u.nm):
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_opd = poppy.StatisticalPSDWFE(index=opd_index, wfe=opd_rms, radius=diam/2, seed=opd_seed).get_opd(wf)
    wfe_amp = poppy.StatisticalPSDWFE(index=amp_index, wfe=amp_rms, radius=diam/2, seed=amp_seed).get_opd(wf)
    wfe_amp /= amp_rms.unit.to(u.m)
    wfe_amp += 1 - amp_rms.to_value(u.m)/amp_rms.unit.to(u.m)/2
    
    wfe_amp = xp.asarray(wfe_amp.get())
    wfe_opd = xp.asarray(wfe_opd.get())
    wfe = wfe_amp * xp.exp(1j*2*np.pi/wavelength.to_value(u.m) * wfe_opd)
    wfe *= xp.asarray(poppy.CircularAperture(radius=diam/2).get_transmission(wf).get())
    
    return wfe

def fft(arr):
    ftarr = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(arr)))
    return ftarr

def ifft(arr):
    iftarr = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(arr)))
    return iftarr

def mft(wavefront, nlamD, npix, forward=True, centering='ADJUSTABLE'):
        '''
        npix : int
            Number of pixels per side side of destination plane array (corresponds
            to 'N_B' in Soummer et al. 2007 4.2). This will be the # of pixels in
            the image plane for a forward transformation, in the pupil plane for an
            inverse.
        '''
        
        # this code was duplicated from POPPY's MFT method
        npupY, npupX = wavefront.shape
        nlamDX, nlamDY = nlamD, nlamD
        npixY, npixX = npix, npix
        
        if forward:
            dU = nlamDX / float(npixX)
            dV = nlamDY / float(npixY)
            dX = 1.0 / float(npupX)
            dY = 1.0 / float(npupY)
        else:
            dX = nlamDX / float(npupX)
            dY = nlamDY / float(npupY)
            dU = 1.0 / float(npixX)
            dV = 1.0 / float(npixY)
        
        if centering=='ADJUSTABLE':
            offsetY, offsetX = 0.0, 0.0
            Xs = (xp.arange(npupX, dtype=float) - float(npupX) / 2.0 - offsetX + 0.5) * dX
            Ys = (xp.arange(npupY, dtype=float) - float(npupY) / 2.0 - offsetY + 0.5) * dY

            Us = (xp.arange(npixX, dtype=float) - float(npixX) / 2.0 - offsetX + 0.5) * dU
            Vs = (xp.arange(npixY, dtype=float) - float(npixY) / 2.0 - offsetY + 0.5) * dV
        elif centering=='FFTSTYLE':
            Xs = (xp.arange(npupX, dtype=float) - (npupX / 2)) * dX
            Ys = (xp.arange(npupY, dtype=float) - (npupY / 2)) * dY

            Us = (xp.arange(npixX, dtype=float) - npixX / 2) * dU
            Vs = (xp.arange(npixY, dtype=float) - npixY / 2) * dV
        
        XU = xp.outer(Xs, Us)
        YV = xp.outer(Ys, Vs)
        
        if forward:
            expXU = xp.exp(-2.0 * np.pi * -1j * XU)
            expYV = xp.exp(-2.0 * np.pi * -1j * YV).T
            t1 = xp.dot(expYV, wavefront)
            t2 = xp.dot(t1, expXU)
        else:
            expYV = xp.exp(-2.0 * np.pi * 1j * YV).T
            expXU = xp.exp(-2.0 * np.pi * 1j * XU)
            t1 = xp.dot(expYV, wavefront)
            t2 = xp.dot(t1, expXU)

        norm_coeff = np.sqrt((nlamDY * nlamDX) / (npupY * npupX * npixY * npixX))
        
        return norm_coeff * t2

class SCOOB():

    def __init__(self, 
                 wavelength=None, 
                 pupil_diam=6.75*u.mm,
                 dm_fill_factor=0.95,
                 npix=256, 
                 oversample=4,
                 npsf=128,
                 psf_pixelscale_lamD=1/5, 
                 detector_rotation=0, 
                 dm_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 Imax_ref=None,
                 WFE=None,
                 FPM=None,
                 LYOT=None):
        
        self.wavelength_c = 632.8e-9*u.m
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.pupil_diam = pupil_diam.to(u.m)
        
        self.npix = npix
        self.oversample = oversample
        self.N = int(npix*oversample)
        
        self.npsf = npsf
        self.psf_pixelscale_lamD = psf_pixelscale_lamD
        self.det_rotation = detector_rotation
        
        self.Imax_ref = Imax_ref
        
        self.WFE = WFE
        self.FPM = FPM
        self.LYOT = LYOT
        
        self.dm_fill_factor = dm_fill_factor # ratio for representing the illuminated area of the DM to accurately compute DM surface
        self.reverse_parity = True
        
        self.init_dm()
        self.init_grids()
        
    def getattr(self, attr):
        return getattr(self, attr)

    def init_dm(self):
        self.Nact = 34
        self.Nacts = 952
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        
        self.full_stroke = 1.5e-6*u.m
        
        self.dm_mask = np.ones((self.Nact,self.Nact), dtype=bool)
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>10.5] = 0 # had to set the threshold to 10.5 instead of 10.2 to include edge actuators
        
        self.DM = custom_dm.DeformableMirror(inf_cube='inf_cube.fits')
        
    def reset_dm(self):
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM.command = dm_command
        
    def add_dm(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM.command += dm_command
        
    def get_dm(self):
        return ensure_np_array(self.DM.command)
    
    def get_dm_surface(self):
        dm_pixelscale = self.dm_fill_factor * self.dm_active_diam/(self.npix*u.pix)
        dm_surf = self.DM.get_surface(pixelscale=dm_pixelscale)
        return dm_surf
    
    def init_grids(self):
        self.pupil_pixelscale = self.pupil_diam.to_value(u.m) / self.npix
        
        x_pp = ( xp.linspace(-self.N/2, self.N/2-1, self.N) + 1/2 ) * self.pupil_pixelscale
        ppx, ppy = xp.meshgrid(x_pp, x_pp)
        ppr = xp.sqrt(ppx**2 + ppy**2)
        ppth = xp.arctan2(ppy,ppx)
        
        self.pupil_grid = xp.array([ppr, ppth])
        
        self.PUPIL = ppr < self.pupil_diam.to_value(u.m)/2
        
        self.focal_pixelscale_lamD = 1/self.oversample
        x_fp = ( xp.linspace(-self.N/2, self.N/2-1, self.N) + 1/2 ) * self.focal_pixelscale_lamD
        fpx, fpy = xp.meshgrid(x_fp, x_fp)
        fpr = xp.sqrt(fpx**2 + fpy**2)
        fpth = xp.arctan2(fpy,fpx)
        
#         self.focal_grid_car = xp.array([fpx, fpy])
        self.focal_grid_pol = xp.array([fpr, fpth])
        
        x_im = ( xp.linspace(-self.npsf/2, self.npsf/2-1, self.npsf) + 1/2 ) * self.psf_pixelscale_lamD
        imx, imy = xp.meshgrid(x_im, x_im)
        imr = xp.sqrt(imx**2 + imy**2)
        imth = xp.arctan2(imy,imx)
        
        self.im_grid_car = xp.array([imx, imy])
        self.im_grid_pol = xp.array([imr, imth])
    
    def apply_dm(self, wavefront, include_reflection=True):
        dm_surf = self.get_dm_surface()
        if include_reflection:
            dm_surf *= 2
        dm_surf = scoobpsf.utils.pad_or_crop(dm_surf, self.N)
        wavefront *= xp.exp(1j*2*np.pi/self.wavelength.to_value(u.m) * dm_surf)
        return wavefront
    
    def propagate(self):
        self.init_grids()
        
        WFE = xp.ones((self.N, self.N), dtype=xp.complex128) if self.WFE is None else self.WFE
        FPM = xp.ones((self.N, self.N), dtype=xp.complex128) if self.FPM is None else self.FPM
        LYOT = xp.ones((self.N, self.N), dtype=xp.complex128) if self.LYOT is None else self.LYOT
        
        self.wavefront = xp.ones((self.N,self.N), dtype=xp.complex128)
        self.wavefront *= self.PUPIL # apply the pupil
        
        self.wavefront *= WFE # apply WFE data
        self.wavefront = self.apply_dm(self.wavefront)# apply the DM
        
        if self.FPM is not None: 
            self.wavefront = fft(self.wavefront)
            self.wavefront *= FPM
            self.wavefront = ifft(self.wavefront)
    
#         imshows.imshow1(xp.abs(self.wavefront))
        self.wavefront *= LYOT # apply the Lyot stop
        
        # propagate to image plane with MFT
        self.nlamD = self.npsf * self.psf_pixelscale_lamD * self.oversample
        self.wavefront = mft(self.wavefront, self.nlamD, self.npsf)
        
        if self.Imax_ref is not None:
            self.wavefront /= xp.sqrt(self.Imax_ref)
        
        if self.reverse_parity:
            self.wavefront = xp.rot90(xp.rot90(self.wavefront))
        
        return self.wavefront
    
    def propagate(self, return_all=False):
        self.init_grids()
        
        if return_all:
            wavefronts = []
        
        WFE = xp.ones((self.N, self.N), dtype=xp.complex128) if self.WFE is None else self.WFE
        FPM = xp.ones((self.N, self.N), dtype=xp.complex128) if self.FPM is None else self.FPM
        LYOT = xp.ones((self.N, self.N), dtype=xp.complex128) if self.LYOT is None else self.LYOT
        
        self.wavefront = xp.ones((self.N,self.N), dtype=xp.complex128)
        self.wavefront *= self.PUPIL # apply the pupil
        if return_all: wavefronts.append(copy.copy(self.wavefront))
        
        self.wavefront *= WFE # apply WFE data
        if return_all: wavefronts.append(copy.copy(self.wavefront))
        
        self.wavefront = self.apply_dm(self.wavefront)# apply the DM
        if return_all: wavefronts.append(copy.copy(self.wavefront))
            
        if self.FPM is not None: 
            self.wavefront = fft(self.wavefront)
            if return_all: wavefronts.append(copy.copy(self.wavefront))
            self.wavefront *= FPM
            if return_all: wavefronts.append(copy.copy(self.wavefront))
            self.wavefront = ifft(self.wavefront)
            if return_all: wavefronts.append(copy.copy(self.wavefront))
        
        self.wavefront *= LYOT # apply the Lyot stop
        if return_all: wavefronts.append(copy.copy(self.wavefront))
            
        # propagate to image plane with MFT
        self.nlamD = self.npsf * self.psf_pixelscale_lamD * self.oversample
        self.wavefront = mft(self.wavefront, self.nlamD, self.npsf)
        
        if self.Imax_ref is not None:
            self.wavefront /= xp.sqrt(self.Imax_ref)
        
        if self.reverse_parity:
            self.wavefront = xp.rot90(xp.rot90(self.wavefront))
            
        if self.det_rotation is not None:
            self.wavefront = self.rotate_wf(self.wavefront)
            
        if return_all: wavefronts.append(copy.copy(self.wavefront))
        
        if return_all:
            return wavefronts
        else:
            return self.wavefront
    
    def rotate_wf(self, wavefront):
        wavefront_r = _scipy.ndimage.rotate(xp.real(wavefront), angle=-self.det_rotation, reshape=False, order=1)
        wavefront_i = _scipy.ndimage.rotate(xp.imag(wavefront), angle=-self.det_rotation, reshape=False, order=1)
        
        new_wavefront = (wavefront_r + 1j*wavefront_i)
        return new_wavefront
    
    def calc_psf(self):
        fpwf = self.propagate(return_all=False)
        return fpwf
    
    def snap(self, plot=False, vmax=None, vmin=None, grid=False):
        fpwf = self.propagate(return_all=False)
        image = xp.abs(fpwf)**2
        if plot:
            imshows.imshow1(ensure_np_array(image), pxscl=self.psf_pixelscale_lamD,
                            lognorm=True, vmax=vmax, vmin=vmin,
                            grid=grid)
        return image



