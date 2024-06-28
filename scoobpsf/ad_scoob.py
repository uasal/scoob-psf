import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path

import poppy

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

from .math_module import xp, _scipy, ensure_np_array
from .imshows import *

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

def make_gaussian_inf_fun(act_spacing=300e-6*u.m, 
                          sampling=25, 
                          Nacts_per_inf=4, # number of influence functions across the grid
                          coupling=0.15,
                          plot=False,
                          save_fits=None):

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
    
    if plot:
        fig,ax = imshow1(ensure_np_array(inf), pxscl=pxscl, 
                         patches=[patches.Circle((0,0), rcoupled, fill=False, color='c'),
                                  patches.Circle((0,0), rcoupled/2, fill=False, color='g', linewidth=1.5)], 
                            display_fig=False, return_fig=True)
        if Nacts_per_inf%2==0:
            ticks = np.linspace(-ext.value/2, ext.value/2, Nacts_per_inf+1)
        else:
            ticks = np.linspace(-ext.value/2, ext.value/2, Nacts_per_inf)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid()
        display(fig)

    if save_fits is not None:
        hdr = fits.Header()
        hdr['SAMPLING'] = sampling
        hdr.comments['SAMPLING'] = '# pixels per actuator'
        hdr['NACTS'] = ext_act
        hdr.comments['NACTS'] = '# actuators across grid'
        inf_hdu = fits.PrimaryHDU(data=ensure_np_array(inf), header=hdr)
        inf_hdu.writeto(str(save_fits), overwrite=True)

    return inf

def get_surface(command, 
                inf_fun, inf_sampling,):

    Nact = command.shape[0]
    inf_sampling = inf_sampling
    inf_fun = inf_fun

    xc = inf_sampling*(xp.linspace(-Nact//2, Nact//2-1, Nact) + 1/2)
    yc = inf_sampling*(xp.linspace(-Nact//2, Nact//2-1, Nact) + 1/2)

    Nsurf = int(inf_sampling*Nact)
    Nsurf = int(2 ** np.ceil(np.log2(Nsurf - 1)))  # next power of 2

    fx = xp.fft.fftfreq(Nsurf)
    fy = xp.fft.fftfreq(Nsurf)

    Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,xc))
    My = xp.exp(-1j*2*np.pi*xp.outer(yc,fy))

    mft_command = Mx@command@My

    fourier_inf_fun = xp.fft.fft2(pad_or_crop(inf_fun, Nsurf))
    fourier_surf = fourier_inf_fun * mft_command
    
    surf = xp.fft.ifft2(fourier_surf).real

    return surf

def reverse_dm(command, inf_fun, inf_sampling,):

    Nact = command.shape[0]
    inf_sampling = inf_sampling
    inf_fun = inf_fun

    xc = inf_sampling*(xp.linspace(-Nact//2, Nact//2-1, Nact) + 1/2)
    yc = inf_sampling*(xp.linspace(-Nact//2, Nact//2-1, Nact) + 1/2)

    Nsurf = int(inf_sampling*Nact)
    Nsurf = int(2 ** np.ceil(np.log2(Nsurf - 1)))  # next power of 2

    fx = xp.fft.fftfreq(Nsurf)
    fy = xp.fft.fftfreq(Nsurf)

    Mx = xp.exp(-1j*2*np.pi*xp.outer(xc,fx))
    My = xp.exp(-1j*2*np.pi*xp.outer(fy,yc))

    mft_command = Mx@command@My

    fourier_inf_fun = xp.fft.fft2(pad_or_crop(inf_fun, Nsurf))
    fourier_surf = fourier_inf_fun * mft_command
    
    surf = xp.fft.ifft2(fourier_surf).real

    return surf

# PROPAGATION FUNCTIONS

def fft(arr):
    return xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(arr)))

def ifft(arr):
    return xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(arr)))

def mft_forward(pupil, psf_pixelscale_lamD, npsf):
    """_summary_

    Parameters
    ----------
    pupil : complex 2D array
        the pupil plane wavefront
    psf_pixelscale_lamD : scalar
        the pixelscale of the desired focal plane wavefront in terms
        of lambda/D
    npsf : integer
        the size of the desired focal plane in pixels

    Returns
    -------
    complex 2D array
        the complex wavefront at the focal plane
    """

    npix = pupil.shape[0]
    dx = 1.0 / npix
    Xs = (xp.arange(npix, dtype=float) - (npix / 2)) * dx

    du = psf_pixelscale_lamD
    Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du

    xu = xp.outer(Us, Xs)
    vy = xp.outer(Xs, Us)

    My = xp.exp(-1j*2*np.pi*vy) 
    Mx = xp.exp(-1j*2*np.pi*xu) 

    norm_coeff = psf_pixelscale_lamD/npix

    return Mx@pupil@My * norm_coeff

# Forward Model
def forward_model(dm_command, WFE, FPM, 
                  wavelength=633e-9, 
                  pixelscale_lamD=1/4, npsf=128, 
                  Imax_ref=1):
        
        
        wavefront = pad_or_crop(APERTURE, N).astype(xp.complex128)
        wavefront *= pad_or_crop(WFE, N) # apply WFE data

        dm_surf = get_surface(dm_command, inf_fun, inf_sampling)
        dm_opd = pad_or_crop(2*dm_surf, N)
        # imshow2(dm_command, dm_surf)
        wavefront *= xp.exp(1j*2*np.pi*dm_opd/wavelength)
        
        wavefront = fft(wavefront)
        wavefront *= FPM
        wavefront = ifft(wavefront)
        imshow2(xp.abs(wavefront), LYOT, npix1=2*npix, npix2=2*npix, lognorm=True)

        wavefront *= pad_or_crop(LYOT, N) # apply the Lyot stop
        imshow1(xp.abs(wavefront), npix=2*npix, lognorm=True)

        wavefront = pad_or_crop(wavefront, npix_lyot)
        imshow1(xp.abs(wavefront), lognorm=True)

        # propagate to image plane with MFT
        # im_oversample = 1/pixelscale_lamD
        # Nim = int(np.floor(npix_lyot*im_oversample))
        # wavefront = fft(wavefront)
        wavefront = mft_forward(wavefront, pixelscale_lamD, npsf)
        
        wavefront /= xp.sqrt(Imax_ref)
        
        return wavefront


    
