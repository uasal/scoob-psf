import numpy as np
import astropy.units as u
from astropy.io import fits
import os
from pathlib import Path
import copy

from skimage.filters import threshold_otsu

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

from .import dm
from .math_module import xp,_scipy, ensure_np_array
from . import imshows
from . import utils

import hcipy
from scipy.signal import windows

def make_vortex_phase_mask(focal_grid_polar, charge=6, 
                           singularity=None, focal_length=500*u.mm, pupil_diam=9.7*u.mm, wavelength=632.8*u.nm):
    
    r = focal_grid_polar[0]
    th = focal_grid_polar[1]
    
    phasor = xp.exp(1j*charge*th)
    
    if singularity is not None:
#         sing*D/(focal_length*lam)
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose().value
        phasor *= mask
    
    return phasor

def fft(arr):
    ftarr = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(arr)))
    return ftarr

def ifft(arr):
    iftarr = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(arr)))
    return iftarr

def mft(wavefront, nlamD, npix, forward=True, centering='ADJUSTABLE'):
        '''
        npix : int
            Number of pixels per side of destination plane array (corresponds
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
                 bad_acts=None,
                 inf_fun=None, # defaults to inf.fits
                 inf_sampling=None,
                 inf_cube=None,
                 Imax_ref=None,
                 WFE=None,
                 FPM=None,
                 LYOT=None,
                 FIELDSTOP=None):
        
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
        self.FIELDSTOP = FIELDSTOP

        self.dm_fill_factor = dm_fill_factor # ratio for representing the illuminated area of the DM to accurately compute DM surface
        self.reverse_parity = True

        self.bad_acts = bad_acts
        self.init_dm()
        self.dm_ref = dm_ref
        self.set_dm(dm_ref)

        self.init_grids()
        
    def getattr(self, attr):
        return getattr(self, attr)

    def init_dm(self):
        self.DM = dm.DeformableMirror()

        self.Nact = self.DM.Nact
        self.Nacts = self.DM.Nacts
        self.act_spacing = self.DM.act_spacing
        self.dm_active_diam = self.DM.active_diam
        self.dm_full_diam = self.DM.pupil_diam
        
        self.full_stroke = self.DM.full_stroke
        
        self.dm_mask = self.DM.dm_mask
        
    def zero_dm(self):
        self.set_dm(xp.zeros((self.Nact,self.Nact)))
    
    def reset_dm(self):
        self.set_dm(self.dm_ref)
    
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
        return self.DM.command
    
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
    
    
    def apply_vortex(self, pupil_wavefront, Nprops=4, window_size=32):

        for i in range(Nprops):
            if i==0: # this is the generic FFT step
                focal_pixelscale_lamD = 1/self.oversample
                x_fp = ( xp.linspace(-self.N/2, self.N/2-1, self.N) + 1/2 ) * focal_pixelscale_lamD
                fpx, fpy = xp.meshgrid(x_fp, x_fp)
                focal_grid = xp.array([xp.sqrt(fpx**2 + fpy**2), xp.arctan2(fpy,fpx)])

                vortex = make_vortex_phase_mask(focal_grid, charge=6, )

                wx = xp.array(windows.tukey(window_size, 1, False))
                wy = xp.array(windows.tukey(window_size, 1, False))
                w = xp.outer(wy, wx)
                w = xp.pad(w, (focal_grid[0].shape[0] - w.shape[0]) // 2, 'constant')
                vortex *= 1 - w
                
                next_pixelscale = w*focal_pixelscale_lamD/self.N # for the next propagation iteration

                # E_LS = E_pup
            else: # this will handle the MFT stages

                mft_pixelscale_lamD = next_pixelscale
                nmft = 128
                nlamD = mft_pixelscale_lamD * nmft

                x_fp = ( xp.linspace(-nmft/2, nmft/2-1, nmft) + 1/2 ) * mft_pixelscale_lamD
                fpx, fpy = xp.meshgrid(x_fp, x_fp)
                focal_grid = xp.array([xp.sqrt(fpx**2 + fpy**2), xp.arctan2(fpy,fpx)])

                vortex = make_vortex_phase_mask(focal_grid, charge=6, )

                wx = xp.array(windows.tukey(window_size, 1, False))
                wy = xp.array(windows.tukey(window_size, 1, False))
                w = xp.outer(wy, wx)
                w = xp.pad(w, (focal_grid[0].shape[0] - w.shape[0]) // 2, 'constant')
                vortex *= 1 - w

                # take the MFT of the pupil_wavefront
                mft(pupil_wavefront, nlamD, nmft, forward=True, centering='ADJUSTABLE')

                # apply the windowed vortex



                # take the inverse MFT to go back to the pupil plane

                # adjust the pixelscale for the next iteration of propagation
                next_pixelscale = mft_pixelscale_lamD/self.N # for the next propagation iteration

                # add the new pupil wavefront to the total pupil wavefront
                # E_LS += E_pup
            
        return

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
        
        if self.FIELDSTOP is not None:
            self.wavefront = fft(self.wavefront)
            if return_all: wavefronts.append(copy.copy(self.wavefront))
            self.wavefront *= self.FIELDSTOP
            if return_all: wavefronts.append(copy.copy(self.wavefront))
            self.wavefront = ifft(self.wavefront)
            if return_all: wavefronts.append(copy.copy(self.wavefront))

        # propagate to image plane with MFT
        self.nlamD = self.npsf * self.psf_pixelscale_lamD * self.oversample
        self.wavefront = mft(self.wavefront, self.nlamD, self.npsf)
        
        if self.Imax_ref is not None:
            self.wavefront /= xp.sqrt(self.Imax_ref)
        
        if self.reverse_parity:
            self.wavefront = xp.rot90(xp.rot90(self.wavefront))
            
        if self.det_rotation is not None:
            self.wavefront = utils.rotate_arr(self.wavefront, rotation=self.det_rotation, order=3)
            
        if return_all: wavefronts.append(copy.copy(self.wavefront))
        
        if return_all:
            return wavefronts
        else:
            return self.wavefront
    
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


