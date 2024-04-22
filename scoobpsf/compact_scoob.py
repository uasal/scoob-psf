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

def make_vortex_phase_mask(focal_grid_polar, 
                           charge=6, 
                           singularity=None, 
                           focal_length=500*u.mm, 
                           pupil_diam=10*u.mm, 
                           wavelength=633*u.nm):
    
    r = focal_grid_polar[0]
    th = focal_grid_polar[1]
    
    phasor = xp.exp(1j*charge*th)
    
    if singularity is not None:
#         sing*D/(focal_length*lam)
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose().value
        phasor *= mask
    
    return phasor

def fft(arr):
    ftarr = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(arr)))
    return ftarr

def ifft(arr):
    iftarr = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.ifftshift(arr)))
    return iftarr

def mft(plane, nlamD, npix, offset=None, inverse=False, centering='FFTSTYLE'):
    """Perform a matrix discrete Fourier transform with selectable
    output sampling and centering.

    Where parameters can be supplied as either scalars or 2-tuples, the first
    element of the 2-tuple is used for the Y dimension and the second for the
    X dimension. This ordering matches that of numpy.ndarray.shape attributes
    and that of Python indexing.

    To achieve exact correspondence to the FFT set nlamD and npix to the size
    of the input array in pixels and use 'FFTSTYLE' centering. (n.b. When
    using `numpy.fft.fft2` you must `numpy.fft.fftshift` the input pupil both
    before and after applying fft2 or else it will introduce a checkerboard
    pattern in the signs of alternating pixels!)

    Parameters
    ----------
    plane : 2D ndarray
        2D array (either real or complex) representing the input image plane or
        pupil plane to transform.
    nlamD : float or 2-tuple of floats (nlamDY, nlamDX)
        Size of desired output region in lambda / D units, assuming that the
        pupil fills the input array (corresponds to 'm' in
        Soummer et al. 2007 4.2). This is in units of the spatial frequency that
        is just Nyquist sampled by the input array.) If given as a tuple,
        interpreted as (nlamDY, nlamDX).
    npix : int or 2-tuple of ints (npixY, npixX)
        Number of pixels per side side of destination plane array (corresponds
        to 'N_B' in Soummer et al. 2007 4.2). This will be the # of pixels in
        the image plane for a forward transformation, in the pupil plane for an
        inverse. If given as a tuple, interpreted as (npixY, npixX).
    inverse : bool, optional
        Is this a forward or inverse transformation? (Default is False,
        implying a forward transformation.)
    centering : {'FFTSTYLE', 'SYMMETRIC', 'ADJUSTABLE'}, optional
        What type of centering convention should be used for this FFT?

        * ADJUSTABLE (the default) For an output array with ODD size n,
          the PSF center will be at the center of pixel (n-1)/2. For an output
          array with EVEN size n, the PSF center will be in the corner between
          pixel (n/2-1, n/2-1) and (n/2, n/2)
        * FFTSTYLE puts the zero-order term in a single pixel.
        * SYMMETRIC spreads the zero-order term evenly between the center
          four pixels

    offset : 2-tuple of floats (offsetY, offsetX)
        For ADJUSTABLE-style transforms, an offset in pixels by which the PSF
        will be displaced from the central pixel (or cross). Given as
        (offsetY, offsetX).
    """
    npupY, npupX = plane.shape

    try:
        if np.isscalar(npix):
            npixY, npixX = float(npix), float(npix)
        else:
            npixY, npixX = tuple(xp.asarray(npix, dtype=float))
    except ValueError:
        raise ValueError(
            "'npix' must be supplied as a scalar (for square arrays) or as "
            "a 2-tuple of ints (npixY, npixX)"
        )

    # make sure these are integer values
    if npixX != int(npixX) or npixY != int(npixY):
        raise TypeError("'npix' must be supplied as integer value(s)")

    try:
        if np.isscalar(nlamD):
            nlamDY, nlamDX = float(nlamD), float(nlamD)
        else:
            nlamDY, nlamDX = tuple(xp.asarray(nlamD, dtype=float))
    except ValueError:
        raise ValueError(
            "'nlamD' must be supplied as a scalar (for square arrays) or as"
            " a 2-tuple of floats (nlamDY, nlamDX)"
        )

    centering = centering.upper()


    # In the following: X and Y are coordinates in the input plane
    #                   U and V are coordinates in the output plane

    if inverse:
        dX = nlamDX / float(npupX)
        dY = nlamDY / float(npupY)
        dU = 1.0 / float(npixX)
        dV = 1.0 / float(npixY)
    else:
        dU = nlamDX / float(npixX)
        dV = nlamDY / float(npixY)
        dX = 1.0 / float(npupX)
        dY = 1.0 / float(npupY)


    if centering == 'FFTSTYLE':
        Xs = (xp.arange(npupX, dtype=float) - (npupX / 2)) * dX
        Ys = (xp.arange(npupY, dtype=float) - (npupY / 2)) * dY

        Us = (xp.arange(npixX, dtype=float) - npixX / 2) * dU
        Vs = (xp.arange(npixY, dtype=float) - npixY / 2) * dV
    elif centering == 'ADJUSTABLE':
        if offset is None:
            offsetY, offsetX = 0.0, 0.0
        else:
            try:
                offsetY, offsetX = tuple(xp.asarray(offset, dtype=float))
            except ValueError:
                raise ValueError(
                    "'offset' must be supplied as a 2-tuple with "
                    "(y_offset, x_offset) as floating point values"
                )
        Xs = (xp.arange(npupX, dtype=float) - float(npupX) / 2.0 - offsetX + 0.5) * dX
        Ys = (xp.arange(npupY, dtype=float) - float(npupY) / 2.0 - offsetY + 0.5) * dY

        Us = (xp.arange(npixX, dtype=float) - float(npixX) / 2.0 - offsetX + 0.5) * dU
        Vs = (xp.arange(npixY, dtype=float) - float(npixY) / 2.0 - offsetY + 0.5) * dV
    elif centering == 'SYMMETRIC':
        Xs = (xp.arange(npupX, dtype=float) - float(npupX) / 2.0 + 0.5) * dX
        Ys = (xp.arange(npupY, dtype=float) - float(npupY) / 2.0 + 0.5) * dY

        Us = (xp.arange(npixX, dtype=float) - float(npixX) / 2.0 + 0.5) * dU
        Vs = (xp.arange(npixY, dtype=float) - float(npixY) / 2.0 + 0.5) * dV
    else:
        raise ValueError("Invalid centering style")

    XU = xp.outer(Xs, Us)
    YV = xp.outer(Ys, Vs)

    # SIGN CONVENTION: plus signs in exponent for basic forward propagation, with
    # phase increasing with time. This convention differs from prior poppy version < 1.0
    if inverse:
        expYV = xp.exp(-2.0 * np.pi * 1j * YV).T
        expXU = xp.exp(-2.0 * np.pi * 1j * XU)
        t1 = xp.dot(expYV, plane)
        t2 = xp.dot(t1, expXU)
    else:
        expXU = xp.exp(-2.0 * np.pi * -1j * XU)
        expYV = xp.exp(-2.0 * np.pi * -1j * YV).T
        t1 = xp.dot(expYV, plane)
        t2 = xp.dot(t1, expXU)

    norm_coeff = np.sqrt((nlamDY * nlamDX) / (npupY * npupX * npixY * npixX))
    return norm_coeff * t2


class SCOOB():

    def __init__(self, 
                 wavelength=None, 
                 npix=256, 
                 oversample=4,
                 npsf=128,
                 psf_pixelscale_lamD=1/5, 
                 detector_rotation=0, 
                 dm_ref=np.zeros((34,34)),
                 bad_acts=None,
                 Imax_ref=None,
                 APERTURE=None, 
                 WFE=None,
                 FPM=None,
                 LYOT=None,
                 FIELDSTOP=None):
        
        self.wavelength_c = 632.8e-9*u.m
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.pupil_diam = 6.75*u.mm

        self.bad_acts = bad_acts
        self.init_dm()
        self.dm_ref = dm_ref
        self.set_dm(dm_ref)

        self.dm_diam = 10.2*u.mm
        self.dm_pupil_diam = 9.2*u.mm
        self.dm_fill_factor = (self.dm_pupil_diam/self.dm_diam).decompose().value

        self.lyot_stop_diam = 8.7*u.mm
        self.lyot_pupil_diam = 9.6*u.mm

        self.lyot_stop_ratio = (self.lyot_stop_diam/self.lyot_pupil_diam).decompose().value
        
        self.npix = npix
        self.oversample = oversample
        self.N = int(npix*oversample)
        
        self.npsf = npsf
        self.psf_pixelscale_lamD = psf_pixelscale_lamD
        self.det_rotation = detector_rotation

        self.Imax_ref = Imax_ref
        
        self.APERTURE = APERTURE
        self.WFE = WFE
        self.FPM = FPM
        self.LYOT = LYOT
        self.FIELDSTOP = FIELDSTOP
        
        self.reverse_parity = True

        self.use_llowfsc = False
        self.llowfsc_defocus = 0.9*u.mm
        self.llowfsc_fl = 100*u.mm
        
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
    
    def apply_dm(self, wavefront, include_reflection=True):
        dm_surf = self.get_dm_surface()
        if include_reflection:
            dm_surf *= 2
        dm_surf = scoobpsf.utils.pad_or_crop(dm_surf, self.N)
        wavefront *= xp.exp(1j*2*np.pi/self.wavelength.to_value(u.m) * dm_surf)
        return wavefront
    
    def make_grid(self, which='pupil', polar=False):
        self.pupil_pixelscale = self.pupil_diam.to_value(u.m) / self.npix
        
        if which=='pupil':
            x, y = utils.make_grid(self.npix, self.pupil_pixelscale)
        elif which=='fpm' or which=='focal':
            x,y = utils.make_grid(self.N, 1/self.oversample)
        elif which=='image' or which=='im':
            x,y = utils.make_grid(self.Npsf, self.psf_pixelscale_lamD)

        if polar:
            r = xp.sqrt(x**2 + y**2)
            th = xp.arctan2(y,x)
            return xp.array([r,th])
        else:
            return xp.array([x,y])
    
    # def apply_vortex(self, pupil_wavefront, Nprops=4, window_size=32):

    #     for i in range(Nprops):
    #         if i==0: # this is the generic FFT step
    #             focal_pixelscale_lamD = 1/self.oversample
    #             x_fp = ( xp.linspace(-self.N/2, self.N/2-1, self.N) + 1/2 ) * focal_pixelscale_lamD
    #             fpx, fpy = xp.meshgrid(x_fp, x_fp)
    #             focal_grid = xp.array([xp.sqrt(fpx**2 + fpy**2), xp.arctan2(fpy,fpx)])

    #             vortex = make_vortex_phase_mask(focal_grid, charge=6, )

    #             wx = xp.array(windows.tukey(window_size, 1, False))
    #             wy = xp.array(windows.tukey(window_size, 1, False))
    #             w = xp.outer(wy, wx)
    #             w = xp.pad(w, (focal_grid[0].shape[0] - w.shape[0]) // 2, 'constant')
    #             vortex *= 1 - w
                
    #             next_pixelscale = w*focal_pixelscale_lamD/self.N # for the next propagation iteration

    #             # E_LS = E_pup
    #         else: # this will handle the MFT stages

    #             mft_pixelscale_lamD = next_pixelscale
    #             nmft = 128
    #             nlamD = mft_pixelscale_lamD * nmft

    #             x_fp = ( xp.linspace(-nmft/2, nmft/2-1, nmft) + 1/2 ) * mft_pixelscale_lamD
    #             fpx, fpy = xp.meshgrid(x_fp, x_fp)
    #             focal_grid = xp.array([xp.sqrt(fpx**2 + fpy**2), xp.arctan2(fpy,fpx)])

    #             vortex = make_vortex_phase_mask(focal_grid, charge=6, )

    #             wx = xp.array(windows.tukey(window_size, 1, False))
    #             wy = xp.array(windows.tukey(window_size, 1, False))
    #             w = xp.outer(wy, wx)
    #             w = xp.pad(w, (focal_grid[0].shape[0] - w.shape[0]) // 2, 'constant')
    #             vortex *= 1 - w

    #             # take the MFT of the pupil_wavefront
    #             mft(pupil_wavefront, nlamD, nmft, forward=True, centering='ADJUSTABLE')

    #             # apply the windowed vortex

    #             # take the inverse MFT to go back to the pupil plane

    #             # adjust the pixelscale for the next iteration of propagation
    #             next_pixelscale = mft_pixelscale_lamD/self.N # for the next propagation iteration

    #             # add the new pupil wavefront to the total pupil wavefront
    #             # E_LS += E_pup
            
    #     return

    def propagate(self, return_all=False):
        
        if return_all:
            wavefronts = []
        
        if self.WFE is None: 
            WFE = xp.ones((self.N, self.N), dtype=xp.complex128)
        else:
            WFE = utils.pad_or_crop(self.WFE, self.N)

        FPM = xp.ones((self.N, self.N), dtype=xp.complex128) if self.FPM is None else self.FPM

        self.wavefront = utils.pad_or_crop(self.APERTURE, self.N).astype(xp.complex128) # apply the pupil
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

        self.wavefront *= utils.pad_or_crop(self.LYOT, self.N).astype(xp.complex128) # apply the Lyot stop
        if return_all: wavefronts.append(copy.copy(self.wavefront))
        
        if self.use_llowfsc:
            
            # Mx,My = ...
            # llowfsc_im = 
            return llowfsc_im

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
    
    def calc_wf(self):
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


