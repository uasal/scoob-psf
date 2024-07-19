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

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2]
    else:
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def make_circ_ap(pupil_diam, npix):
    wf = poppy.FresnelWavefront(beam_radius=pupil_diam/2, npix=npix, oversample=1) # pupil wavefront
    ap = poppy.CircularAperture(radius=pupil_diam/2).get_transmission(wf)
    return ap

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

def make_vortex_masks(npix=1000, 
                      oversample_vortex=4.096,
                      plot=False):
    N_vortex_lres = int(npix*oversample_vortex)
    lres_sampling = 1/oversample_vortex # low resolution sampling in lam/D per pixel
    lres_win_size = int(30/lres_sampling) + 1
    w1d = xp.array(windows.tukey(lres_win_size, 1, False))
    lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), N_vortex_lres)
    vortex_lres = props.make_vortex_phase_mask(N_vortex_lres)
    if plot: imshow2(xp.angle(vortex_lres), 1-lres_window, npix1=64, npix2=lres_win_size, pxscl2=lres_sampling)

    hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
    N_vortex_hres = int(np.round(30.029296875/hres_sampling))
    hres_win_size = int(30.029296875/hres_sampling)
    w1d = xp.array(windows.tukey(hres_win_size, 1, False))
    hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), N_vortex_hres)
    vortex_hres = props.make_vortex_phase_mask(N_vortex_hres)

    x = (xp.linspace(-N_vortex_hres//2, N_vortex_hres//2-1, N_vortex_hres)) * hres_sampling
    x,y = xp.meshgrid(x,x)
    r = xp.sqrt(x**2 + y**2)
    sing_mask = r>=0.15
    hres_window *= sing_mask
    if plot: imshow2(xp.angle(vortex_hres), hres_window, npix1=64, npix2=hres_win_size, pxscl2=hres_sampling)

    return vortex_lres, vortex_hres, lres_window, hres_window

def create_control_mask(npsf, psf_pixelscale, iwa=3, owa=12, edge=None, rotation=0):
    x = (xp.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*psf_pixelscale
    x,y = xp.meshgrid(x,x)
    r = xp.hypot(x, y)
    control_mask = (r < owa) * (r > iwa)
    if edge is not None: control_mask *= (x > edge)

    control_mask = _scipy.ndimage.rotate(control_mask, rotation, reshape=False, order=0)
    return control_mask

def forward_model(command, vortex_params=None, use_wfe=True, return_pupil=False):
    Nsurf

    
    dm_surf = inf_matrix.dot(xp.array(actuators)).reshape(Nsurf,Nsurf)
    dm_phasor = xp.exp(1j * 4*xp.pi/wavelength.to_value(u.m) * dm_surf)

    wf = utils.pad_or_crop(APERTURE, N).astype(xp.complex128)
    wf *= utils.pad_or_crop(dm_phasor, N)
    # imshow2(xp.abs(wf), xp.angle(wf), npix=npix)

    if use_wfe: 
        wf *= utils.pad_or_crop(WFE, N)
        # imshow2(xp.abs(wf), xp.angle(wf), npix=npix)

    if return_pupil:
        E_pup = copy.copy(wf)

    if use_vortex:
        lres_wf = utils.pad_or_crop(wf, N_vortex_lres) # pad to the larger array for the low res propagation
        fp_wf_lres = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(lres_wf))) # to FPM
        fp_wf_lres *= vortex_lres * (1 - lres_window) # apply low res (windowed) FPM
        pupil_wf_lres = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fp_wf_lres))) # to Lyot Pupil
        # pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, N)

        hres_wf = utils.pad_or_crop(wf, npix) # crop to the pupil diameter for the high res propagation
        fp_wf_hres = props.mft_forward(hres_wf, hres_sampling, N_vortex_hres)
        fp_wf_hres *= vortex_hres * hres_window # apply high res (windowed) FPM
        # pupil_wf_hres = props.mft_reverse(fp_wf_hres, hres_sampling, npix,)
        # pupil_wf_hres = utils.pad_or_crop(pupil_wf_hres, N)
        pupil_wf_hres = props.mft_reverse(fp_wf_hres, hres_sampling*oversample_vortex, N_vortex_lres,)

        wf = (pupil_wf_lres + pupil_wf_hres)
        wf = utils.pad_or_crop(wf, N)
        # imshow2(xp.abs(wf), xp.angle(wf))

    wf *= utils.pad_or_crop(LYOT, N)
    # imshow2(xp.abs(wf), xp.angle(wf), npix=2*npix)

    wf = utils.pad_or_crop(wf, nlyot)
    fpwf = props.mft_forward(wf, psf_pixelscale_lamD, npsf)

    if return_pupil:
        return fpwf, E_pup
    else:
        return fpwf





