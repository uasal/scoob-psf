from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows
from . import dm

# from scoobpsf import dm

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

import poppy

from scipy.signal import windows

def fft(arr):
    return xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(arr)))

def ifft(arr):
    return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(arr)))


def ang_spec(wavefront, wavelength, distance, pixelscale):
    """Propagate a wavefront a given distance via the angular spectrum method. 

    Parameters
    ----------
    wavefront : complex 2D array
        the input wavefront
    wavelength : astropy quantity
        the wavelength of the wavefront
    distance : astropy quantity
        distance to propagate wavefront
    pixelscale : astropy quantity
        pixelscale in physical units of the wavefront

    Returns
    -------
    complex 2D array
        the propagated wavefront
    """
    n = wavefront.shape[0]

    delkx = 2*np.pi/(n*pixelscale.to_value(u.m/u.pix))
    kxy = (xp.linspace(-n/2, n/2-1, n) + 1/2)*delkx
    k = 2*np.pi/wavelength.to_value(u.m)
    kx, ky = xp.meshgrid(kxy,kxy)

    wf_as = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(wavefront)))
    
    kz = xp.sqrt(k**2 - kx**2 - ky**2 + 0j)
    tf = xp.exp(1j*kz*distance.to_value(u.m))

    prop_wf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(wf_as * tf)))
    kz = 0.0
    tf = 0.0

    return prop_wf

def mft_forward(wavefront, npix, npsf, psf_pixelscale_lamD, convention='-', pp_centering='even', fp_centering='odd'):
    N = wavefront.shape[0]
    dx = 1.0 / npix
    if pp_centering=='even':
        Xs = (xp.arange(N, dtype=float) - (N / 2) + 1/2) * dx
    elif pp_centering=='odd':
        Xs = (xp.arange(N, dtype=float) - (N / 2)) * dx

    du = psf_pixelscale_lamD
    if fp_centering=='odd':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du
    elif fp_centering=='even':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2 + 1/2) * du

    xu = xp.outer(Us, Xs)
    vy = xp.outer(Xs, Us)

    if convention=='-':
        My = xp.exp(-1j*2*np.pi*vy) 
        Mx = xp.exp(-1j*2*np.pi*xu)
    else:
        My = xp.exp(1j*2*np.pi*vy) 
        Mx = xp.exp(1j*2*np.pi*xu)

    norm_coeff = psf_pixelscale_lamD/npix

    return Mx@wavefront@My * norm_coeff

def mft_reverse(fpwf, psf_pixelscale_lamD, npix, N, convention='+', pp_centering='even', fp_centering='odd'):
    npsf = fpwf.shape[0]
    du = psf_pixelscale_lamD
    if fp_centering=='odd':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du
    elif fp_centering=='even':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2 + 1/2) * du

    dx = 1.0 / npix
    if pp_centering=='even':
        Xs = (xp.arange(N, dtype=float) - (N / 2) + 1/2) * dx
    elif pp_centering=='odd':
        Xs = (xp.arange(N, dtype=float) - (N / 2)) * dx

    ux = xp.outer(Xs, Us)
    yv = xp.outer(Us, Xs)

    if convention=='+':
        My = xp.exp(1j*2*np.pi*yv) 
        Mx = xp.exp(1j*2*np.pi*ux)
    else:
        My = xp.exp(-1j*2*np.pi*yv) 
        Mx = xp.exp(-1j*2*np.pi*ux)

    norm_coeff = psf_pixelscale_lamD/npix 

    return Mx@fpwf@My * norm_coeff

def get_scaled_coords(N, scale, center=True, shift=True):
    if center:
        cen = (N-1)/2.0
    else:
        cen = 0
        
    if shift:
        shiftfunc = xp.fft.fftshift
    else:
        shiftfunc = lambda x: x
    cy, cx = (shiftfunc(xp.indices((N,N))) - cen) * scale
    r = xp.sqrt(cy**2 + cx**2)
    return [cy, cx, r]

def get_fresnel_TF(dz, N, wavelength, fnum):
    '''
    Get the Fresnel transfer function for a shift dz from focus
    '''
    df = 1.0 / (N * wavelength * fnum)
    rp = get_scaled_coords(N,df, shift=False)[-1]
    return xp.exp(-1j*np.pi*dz*wavelength*(rp**2))

def make_vortex_phase_mask(npix, charge=6, 
                           grid='odd', 
                           singularity=None, 
                           focal_length=500*u.mm, pupil_diam=9.5*u.mm, wavelength=650*u.nm):
    
    if grid=='odd':
        x = xp.linspace(-npix//2, npix//2-1, npix)
    elif grid=='even':
        x = xp.linspace(-npix//2, npix//2-1, npix) + 1/2
    x,y = xp.meshgrid(x,x)
    th = xp.arctan2(y,x)

    phasor = xp.exp(1j*charge*th)
    
    if singularity is not None:
        r = xp.sqrt((x-1/2)**2 + (y-1/2)**2)
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose().value
        phasor *= mask
    
    return phasor

def apply_vortex(pupil_wf, npix, plot=False):
    # course FPM first
    Nfpm = pupil_wf.shape[0]
    oversample = Nfpm/npix

    vortex_mask = make_vortex_phase_mask(Nfpm, )
    low_res_sampling = 1/oversample # lam/D per pixel
    window_size = int(30/low_res_sampling)
    w1d = xp.array(windows.tukey(window_size, 1, False))
    low_res_window = 1 - utils.pad_or_crop(xp.outer(w1d, w1d), Nfpm)
    # low_res_window = 1 - _scipy.ndimage.shift(utils.pad_or_crop(xp.outer(w1d, w1d), Nfpm), (1,1))
    if plot: imshows.imshow1(low_res_window, npix=128, pxscl=npix/Nfpm)

    fp_wf_low_res = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(utils.pad_or_crop(pupil_wf, Nfpm)))) # to FPM
    fp_wf_low_res *= vortex_mask * low_res_window # apply FPM
    pupil_wf_low_res = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fp_wf_low_res))) # to Lyot Pupil

    # high res FPM second
    high_res_sampling = 0.025 # lam/D per pixel
    Nmft = int(np.round(30/high_res_sampling))
    vortex_mask = make_vortex_phase_mask(Nmft, )
    window_size = int(30/high_res_sampling)
    w1d = xp.array(windows.tukey(window_size, 1, False))
    high_res_window = utils.pad_or_crop(xp.outer(w1d, w1d), Nmft)

    # x = xp.linspace(-self.Nfpm//2, self.Nfpm//2-1, self.Nfpm) * high_res_sampling
    x = (xp.linspace(-Nmft//2, Nmft//2-1, Nmft)) * high_res_sampling
    x,y = xp.meshgrid(x,x)
    r = xp.sqrt(x**2 + y**2)
    sing_mask = r>0.15
    high_res_window *= sing_mask

    if plot: imshows.imshow1(high_res_window, npix=int(np.round(128*9.765625)), pxscl=high_res_sampling)

    # fp_wf_high_res = mft_forward(utils.pad_or_crop(pupil_wf, npix), high_res_sampling, Nmft)
    # fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
    # pupil_wf_high_res = mft_reverse(fp_wf_high_res, high_res_sampling, npix,)
    # pupil_wf_high_res = utils.pad_or_crop(pupil_wf_high_res, N)
    
    fp_wf_high_res = mft_forward(pupil_wf, high_res_sampling * oversample, Nmft)
    fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
    pupil_wf_high_res = mft_reverse(fp_wf_high_res, high_res_sampling * oversample, Nfpm,)

    post_fpm_pupil = (pupil_wf_low_res + pupil_wf_high_res)

    return post_fpm_pupil


