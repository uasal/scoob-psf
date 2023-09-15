import numpy as np
import cupy as cp
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import ray

from .math_module import xp,_scipy, ensure_np_array
from . import imshows
import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as xp
platform = jax.devices()[0].platform
device = jax.devices()[0].device_kind

print(f'Jax platform: {platform}')
print(f'Jax device: {device}')

'''
FIXME: This file will eventually contain a compact model of SCOOB similar to how FALCO uses a compact model to compute Jacobians
'''

def make_vortex_phase_mask(focal_grid_pol, singularity):
    
    r = focal_grid_pol[0]
    th = focal_grid_pol[1]
    
    mask = r>singularity.to_value(u.m)
    
    phasor = mask*xp.exp(1j*charge*th)
    
    return vortex_phasor

def fft(arr):
    ftarr = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(arr)))
    return ftarr

def ifft(arr):
    iftarr = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(arr)))
    return iftarr

def mft(arr):
    
    return mftarr

class SCOOB():

    def __init__(self, 
                 wavelength=None, 
                 pupil_diam=6.75*u.mm,
                 npix=256, 
                 oversample=4,
                 npsf=100,
#                  psf_pixelscale=5e-6*u.m/u.pix,
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
        
        self.Imax_ref = Imax_ref
        
        self.dm_inf = 'inf.fits' if dm_inf is None else dm_inf
        
        self.WFE = WFE
        self.FPM = FPM
        self.LYOT = LYOT
        
        self.init_dm()
        self.init_grids()
        
        self.det_rotation = detector_rotation
        
        
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
        
        self.dm_zernikes = ensure_np_array(poppy.zernike.arbitrary_basis(xp.array(self.dm_mask), nterms=15, outside=0))
        
        self.DM = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM', 
                                                   actuator_spacing=self.act_spacing, 
                                                   influence_func=self.dm_inf,
                                                  )
        
    def reset_dm(self):
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm(self, dm_command):
        self.DM.set_surface(dm_command)
        
    def add_dm(self, dm_command):
        self.DM.set_surface(self.get_dm() + dm_command)
        
    def get_dm(self):
        return self.DM.surface.get()
    
    def show_dm(self):
        wf = poppy.FresnelWavefront(beam_radius=self.dm_active_diam/2, npix=self.npix, oversample=1)
        misc.imshow2(self.get_dm(), self.DM.get_opd(wf), 'DM Command', 'DM Surface',
                     pxscl2=wf.pixelscale.to(u.mm/u.pix))
    
    def init_grids(self):
        self.pupil_pixelscale = self.pupil_diam.to_value(u.m) / self.npix
        self.N = int(self.npix*self.oversample)
        x_p = ( xp.linspace(-self.N/2, self.N/2-1, self.N) + 1/2 ) * self.pupil_pixelscale
        self.ppx, self.ppy = xp.meshgrid(x_p, x_p)
        self.ppr = xp.sqrt(self.ppx**2 + self.ppy**2)
        
        self.PUPIL = self.ppr < self.pupil_diam.to_value(u.m)/2
        
        self.focal_pixelscale_lamD = 1/self.oversample
        x_f = ( xp.linspace(-self.N/2, self.N/2-1, self.N) + 1/2 ) * self.focal_pixelscale_lamD
        fpx, fpy = xp.meshgrid(x_f, x_f)
        fpr = xp.sqrt(fpx**2 + fpy**2)
        fpth = xp.arctan2(fpy,fpx)
        
        self.focal_grid_car = xp.array([self.fpx, self.fpy])
        self.focal_grid_pol = xp.array([self.fpr, self.fpth])
        
        x_im = ( xp.linspace(-self.npsf/2, self.npsf/2-1, self.npsf) + 1/2 ) * self.psf_pixelscale_lamD
        self.imx, self.imy = xp.meshgrid(x_im, x_im)
        self.imr = xp.sqrt(self.imx**2 + self.imy**2)
    
    def apply_dm(self, wavefront):
        fwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=self.oversample)
        dm_opd = self.DM.get_opd(fwf)
        wf_opd = xp.angle(wavefront)*self.wavelength.to_value(u.m)/(2*np.pi)
        wf_opd += dm_opd
        wavefront = xp.abs(wavefront) * xp.exp(1j*2*np.pi/self.wavelength.to_value(u.m) * wf_opd)
        return wavefront
    
    def fft(self, wavefront, forward=True):
        if forward:
            wavefront = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(wavefront)))
        else:
            wavefront = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(wavefront)))
            
        return wavefront
    
    def mft(self, wavefront, nlamD, npix, forward=True, centering='ADJUSTABLE'):
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
    
    def propagate(self):
        self.init_grids()
        
        WFE = xp.ones((self.N, self.N), dtype=xp.complex128) if self.WFE is None else self.WFE
        FPM = xp.ones((self.N, self.N), dtype=xp.complex128) if self.FPM is None else self.FPM
        LYOT = xp.ones((self.N, self.N), dtype=xp.complex128) if self.LYOT is None else self.LYOT
        
        self.wavefront = xp.ones((self.N,self.N), dtype=xp.complex128)
        self.wavefront *= self.PUPIL # apply the pupil
        self.wavefront /= np.float64(xp.sqrt(xp.sum(self.PUPIL))) if self.norm is None else self.norm
        self.wavefront = self.apply_dm(self.wavefront)# apply the DM
        
        # propagate to intermediate focal plane
        self.wavefront = self.fft(self.wavefront)
        
        # propagate to the pre-FPM pupil plane
        self.wavefront = self.fft(self.wavefront, forward=False)
        self.wavefront *= WFE # apply WFE data
        
        if self.FPM is not None: 
#             self.wavefront = self.simple_vortex(self.wavefront)
#             self.wavefront = self.apply_vortex(self.wavefront)
#             in_val, out_val = (0.5, 6)
#             self.wavefront = falco.prop.mft_p2v2p(ensure_np_array(self.wavefront), self.CHARGE, self.npix/2, in_val, out_val)
#             self.wavefront = xp.array(self.wavefront)
            self.wavefront = self.fft(self.wavefront, forward=True)
            self.wavefront *= FPM
            self.wavefront = self.fft(self.wavefront, forward=False)
    
        self.wavefront *= LYOT # apply the Lyot stop
        
        # propagate to image plane with MFT
        self.nlamD = self.npsf * self.psf_pixelscale_lamD * self.oversample
        self.wavefront = self.mft(self.wavefront, self.nlamD, self.npsf)
        
        return self.wavefront
    


