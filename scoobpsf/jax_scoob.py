import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import ray

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

from . import imshows
from . import jax_dm

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
platform = jax.devices()[0].platform
device = jax.devices()[0].device_kind

print(f'Jax platform: {platform}')
print(f'Jax device: {device}')

'''
FIXME: This file will eventually contain a compact model of SCOOB similar to how FALCO uses a compact model to compute Jacobians
'''

# MISCELLANEOUS FUNCTIONS
def ensure_np_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, jax.numpy.ndarray):
        return np.asarray(arr)
    elif isinstance(arr, cp.ndarray):
        return arr.get()

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2]
    else:
        arr_out = jnp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out = arr_out.at[x1:x2,x1:x2].set(arr_in)
    return arr_out

# MAKE OPTICAL ELEMENT FUNCTIONS

def make_pupil(pupil_diam, npix, oversample):
    
    return pupil

def make_vortex_phase_mask(npix, oversample, 
                           charge=6, 
                           singularity=None, 
                           focal_length=500*u.mm, pupil_diam=9.7*u.mm, wavelength=632.8*u.nm):
    
    N = int(npix*oversample)
    
    r = focal_grid_polar[0]
    th = focal_grid_polar[1]
    
    phasor = jnp.exp(1j*charge*th)
    
    if singularity is not None:
#         sing*D/(focal_length*lam)
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose()
        phasor *= mask
    
    return phasor

import poppy
def generate_wfe(diam, 
                 opd_index=2.5, amp_index=2, 
                 opd_seed=1234, amp_seed=12345,
                 opd_rms=10*u.nm, amp_rms=0.05*u.nm,
                 npix=256, oversample=4, 
                 wavelength=500*u.nm):
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_opd = poppy.StatisticalPSDWFE(index=opd_index, wfe=opd_rms, radius=diam/2, seed=opd_seed).get_opd(wf)
    wfe_amp = poppy.StatisticalPSDWFE(index=amp_index, wfe=amp_rms, radius=diam/2, seed=amp_seed).get_opd(wf)
    wfe_amp /= amp_rms.unit.to(u.m)
    wfe_amp += 1 - amp_rms.to_value(u.m)/amp_rms.unit.to(u.m)
    
    wfe_amp = jnp.asarray(wfe_amp.get())
    wfe_opd = jnp.asarray(wfe_opd.get())
    wfe = wfe_amp * jnp.exp(1j*2*np.pi/wavelength.to_value(u.m) * wfe_opd)
    wfe *= jnp.asarray(poppy.CircularAperture(radius=diam/2).get_transmission(wf).get())
    
    return wfe

# PROPAGATION FUNCTIONS

def fft(arr):
    ftarr = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(arr)))
    return ftarr

def ifft(arr):
    iftarr = jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fftshift(arr)))
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
            Xs = (jnp.arange(npupX, dtype=float) - float(npupX) / 2.0 - offsetX + 0.5) * dX
            Ys = (jnp.arange(npupY, dtype=float) - float(npupY) / 2.0 - offsetY + 0.5) * dY

            Us = (jnp.arange(npixX, dtype=float) - float(npixX) / 2.0 - offsetX + 0.5) * dU
            Vs = (jnp.arange(npixY, dtype=float) - float(npixY) / 2.0 - offsetY + 0.5) * dV
        elif centering=='FFTSTYLE':
            Xs = (jnp.arange(npupX, dtype=float) - (npupX / 2)) * dX
            Ys = (jnp.arange(npupY, dtype=float) - (npupY / 2)) * dY

            Us = (jnp.arange(npixX, dtype=float) - npixX / 2) * dU
            Vs = (jnp.arange(npixY, dtype=float) - npixY / 2) * dV
        
        XU = jnp.outer(Xs, Us)
        YV = jnp.outer(Ys, Vs)
        
        if forward:
            expXU = jnp.exp(-2.0 * np.pi * -1j * XU)
            expYV = jnp.exp(-2.0 * np.pi * -1j * YV).T
            t1 = jnp.dot(expYV, wavefront)
            t2 = jnp.dot(t1, expXU)
        else:
            expYV = jnp.exp(-2.0 * np.pi * 1j * YV).T
            expXU = jnp.exp(-2.0 * np.pi * 1j * XU)
            t1 = jnp.dot(expYV, wavefront)
            t2 = jnp.dot(t1, expXU)

        norm_coeff = np.sqrt((nlamDY * nlamDX) / (npupY * npupX * npixY * npixX))
        
        return norm_coeff * t2
    
    
# DM Functions    

def apply_dm(self, wavefront, include_reflection=True):
    dm_surf = self.get_dm_surface()
    if include_reflection:
        dm_surf *= 2
    dm_surf = pad_or_crop(dm_surf, self.N)
    wavefront *= jnp.exp(1j*2*np.pi/self.wavelength.to_value(u.m) * dm_surf)
    return wavefront
    

def forward_model(wavefront, WFE, FPM, LYOT, oversample, npsf=200, pixelscale_lamD=1/5, Imax_ref=1):
        wavefront *= WFE # apply WFE data
#         self.wavefront = self.apply_dm(self.wavefront)# apply the DM
         
        wavefront = fft(self.wavefront)
        wavefront *= FPM
        wavefront = ifft(self.wavefront)
    
        wavefront *= LYOT # apply the Lyot stop
        
        # propagate to image plane with MFT
        nlamD = npsf * pixelscale_lamD * oversample
        wavefront = mft(wavefront, nlamD, npsf)
        
        wavefront /= jnp.sqrt(Imax_ref)
        
        return wavefront

    
def snap(self, plot=False, vmax=None, vmin=None, grid=False):
        fpwf = forward_model()
        image = xp.abs(fpwf)**2
        if plot:
            imshows.imshow1(ensure_np_array(image), pxscl=self.psf_pixelscale_lamD,
                            lognorm=True, vmax=vmax, vmin=vmin,
                            grid=grid)
        return image
    
    
