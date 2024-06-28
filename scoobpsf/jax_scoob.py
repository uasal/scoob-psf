import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path

import poppy

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

from .math_module import cupy_avail, jax_avail, ensure_np_array
from .imshows import *
from . import jax_dm

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
platform = jax.devices()[0].platform
device = jax.devices()[0].device_kind

print(f'Jax platform: {platform}')
print(f'Jax device: {device}')

wavelength_c = 633e-9*u.m
pupil_diam = 6.75*u.mm
dm_pupil_diam = 9.4*u.mm
lyot_pupil_diam = 9.4*u.mm
lyot_diam = 8.6*u.mm
lyot_ratio = 8.6/9.4

psf_pixelscale_lamD = 0.17027909426013688 # at the central wavelength of 633nm

npix = 500
oversample = 8192/npix
N = int(npix*oversample)
npix_lyot = int(lyot_ratio * npix) # 90% diameter lyot stop

Nact = 34
Nacts = 952
dm_mask = jax_dm.make_dm_mask()

wf = poppy.FresnelWavefront(beam_radius=dm_pupil_diam/2, npix=npix, oversample=1) # pupil wavefront
APERTURE = jnp.array(ensure_np_array(poppy.CircularAperture(radius=dm_pupil_diam/2).get_transmission(wf)))

wf = poppy.FresnelWavefront(beam_radius=lyot_pupil_diam/2, npix=npix, oversample=1) # pupil wavefront
LYOT = jnp.array(ensure_np_array(poppy.CircularAperture(radius=lyot_diam/2).get_transmission(wf)))

y,x = (jnp.indices((N, N)) - N//2) * 1/oversample
r = jnp.sqrt(x**2 + y**2)
th = jnp.arctan2(y,x)
opaque_spot = r>0.5
VORTEX = opaque_spot * jnp.exp(1j*6*th)

act_spacing = 300e-6*u.m
pupil_pxscl = dm_pupil_diam.to_value(u.m)/npix
sampling = act_spacing.to_value(u.m)/pupil_pxscl
inf_fun, inf_sampling, inf_pixelscale = jax_dm.make_gaussian_inf_fun(sampling=sampling, 
                                                                     Nacts_per_inf=4, 
                                                                     coupling=0.15,
                                                                    #  plot=True,
                                                                     )

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

def interp_arr(arr, pixelscale, new_pixelscale, order=1):
        Nold = arr.shape[0]
        old_xmax = pixelscale * Nold/2

        x,y = jnp.ogrid[-old_xmax:old_xmax-pixelscale:Nold*1j,
                       -old_xmax:old_xmax-pixelscale:Nold*1j]

        Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) - 1
        new_xmax = new_pixelscale * Nnew/2

        newx,newy = jnp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                              -new_xmax:new_xmax-new_pixelscale:Nnew*1j]

        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0] - x0
        dy = y[0,1] - y0

        ivals = (newx - x0)/dx
        jvals = (newy - y0)/dy

        coords = jnp.array([ivals, jvals])

        interped_arr = jax.scipy.ndimage.map_coordinates(arr, coords, order=order)
        return interped_arr

# MAKE OPTICAL ELEMENT FUNCTIONS

def lstsq(modes, data):
    """Least-Squares fit of modes to data.

    Parameters
    ----------
    modes : iterable
        modes to fit; sequence of ndarray of shape (m, n)
    data : numpy.ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    numpy.ndarray
        fit coefficients

    """
    mask = jnp.isfinite(data)
    data = data[mask]
    modes = jnp.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    modes = modes[:, mask.ravel()].T  # transpose moves modes to columns, as needed for least squares fit
    c, *_ = jnp.linalg.lstsq(modes, data, rcond=None)
    return c


def make_focal_grid(npix, oversample, polar=True, center='pixel'):
    
    N = int(npix*oversample)
    pxscl = 1/oversample
    
    if center=='pixel':
        xx = jnp.linspace(-N/2, N/2-1, N)*pxscl
    elif center=='corner':
        xx = (jnp.linspace(0, N-1, N) - N/2 + 1/2)*pxscl
    print(xx)
    x,y = jnp.meshgrid(xx,xx)

    if polar:
        r = jnp.sqrt(x**2 + y**2)
        th = jnp.arctan2(y,x)
        focal_grid = jnp.array([r,th])
    else:
        focal_grid = jnp.array([x,y])

    return focal_grid

def make_vortex_phase_mask(focal_grid_polar,
                           charge=6, 
                           singularity=None, 
                           focal_length=500*u.mm, pupil_diam=9.6*u.mm, wavelength=633*u.nm):
    
    r = focal_grid_polar[0]
    th = focal_grid_polar[1]
    
    N = th.shape[0]

    phasor = jnp.exp(1j*charge*th)
    
    if singularity is not None:
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose()
        phasor *= mask
    
    return phasor

def generate_wfe(diam, 
                 opd_index=2.5, amp_index=2.5, 
                 opd_seed=1234, amp_seed=12345,
                 opd_rms=10*u.nm, amp_rms=0.05,
                 npix=256,
                 wavelength=500*u.nm):
    amp_rms *= u.nm
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_opd = poppy.StatisticalPSDWFE(index=opd_index, wfe=opd_rms, radius=diam/2, seed=opd_seed).get_opd(wf)
    wfe_amp = poppy.StatisticalPSDWFE(index=amp_index, wfe=amp_rms, radius=diam/2, seed=amp_seed).get_opd(wf)
    wfe_amp /= amp_rms.unit.to(u.m)
    
    wfe_amp = jnp.asarray(ensure_np_array(wfe_amp))
    wfe_opd = jnp.asarray(ensure_np_array(wfe_opd))

    mask = ensure_np_array(poppy.CircularAperture(radius=diam/2).get_transmission(wf))>0
    Zs = ensure_np_array(poppy.zernike.arbitrary_basis(mask, nterms=3, outside=0))
    
    Zc_amp = lstsq(Zs, wfe_amp)
    Zc_opd = lstsq(Zs, wfe_opd)
    for i in range(3):
        wfe_amp -= Zc_amp[i] * Zs[i]
        wfe_opd -= Zc_opd[i] * Zs[i]
    wfe_amp += 1

    wfe = wfe_amp * jnp.exp(1j*2*np.pi/wavelength.to_value(u.m) * wfe_opd)
    wfe *= jnp.asarray(ensure_np_array(poppy.CircularAperture(radius=diam/2).get_transmission(wf)))
    
    return wfe

def make_pupil(pupil_diam, npix, oversample, ratio=1):
    wf = poppy.FresnelWavefront(beam_radius=pupil_diam/2, npix=npix, oversample=oversample)
    circ = poppy.CircularAperture(radius=ratio*pupil_diam/2)
    pupil = ensure_np_array(circ.get_transmission(wf))
    return jnp.array(pupil)

# PROPAGATION FUNCTIONS

def fft(arr):
    return jnp.fft.ifftshift(jnp.fft.fft2(jnp.fft.fftshift(arr)))

def ifft(arr):
    return jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fftshift(arr)))

# Forward Model
def forward_model(dm_command, WFE, FPM, 
                  wavelength=633e-9, 
                  pixelscale_lamD=1/4, 
                  Imax_ref=1):
        
        wavefront = pad_or_crop(APERTURE, N)
        wavefront *= pad_or_crop(WFE, N) # apply WFE data

        dm_surf = jax_dm.get_surf(dm_command, inf_fun, inf_sampling)
        dm_opd = pad_or_crop(2*dm_surf, N)
        imshow2(dm_command, dm_surf)
        wavefront *= jnp.exp(1j*2*np.pi*dm_opd/wavelength)
        
        wavefront = fft(wavefront)
        wavefront *= FPM
        wavefront = ifft(wavefront)
    
        wavefront *= pad_or_crop(LYOT, N) # apply the Lyot stop
        
        # propagate to image plane with MFT
        im_oversample = 1/pixelscale_lamD
        Nim = int(np.ceil(npix_lyot*im_oversample))
        wavefront = pad_or_crop(wavefront, Nim)
        wavefront = fft(wavefront)
        
        wavefront /= jnp.sqrt(Imax_ref)
        
        return wavefront

def get_dm_phasor(command, 
                  inf_fun, inf_sampling, inf_pixelscale,
                  npix, dm_fill_factor,
                  dm_active_diam=10.2*u.mm,
                  wavelength=650e-9*u.m):
    
    pixelscale = dm_fill_factor * dm_active_diam/(npix*u.pix)

    phasor = jax_dm.get_phasor(command, 
                               inf_fun, inf_sampling, 
                               inf_pixelscale=inf_pixelscale, 
                               pixelscale=pixelscale,
                               wavelength=wavelength)

    return phasor

def snap(wavefront, WFE, dm_phasor, FPM, LYOT,
        npix, oversample, 
        npsf=200, pixelscale_lamD=1/5, 
        Imax_ref=1,
         plot=False, vmax=None, vmin=None, grid=False):
        fpwf = forward_model(wavefront, WFE, dm_phasor, FPM, LYOT,
                             npix, oversample, 
                             npsf=npsf, pixelscale_lamD=pixelscale_lamD, 
                             Imax_ref=Imax_ref)
        image = jnp.abs(fpwf)**2
        if plot:
            imshow1(image, pxscl=pixelscale_lamD,
                            lognorm=True, vmax=vmax, vmin=vmin,
                            grid=grid)
        return image
    
    
