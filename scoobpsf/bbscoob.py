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

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image


class BBSCOOBM():
    '''
    This is a class that sets up the parallelization of calc_psf such that it 
    we can generate polychromatic wavefronts that are then fed into the 
    various wavefront simulations.
    '''
    def __init__(self, 
                 wavelengths,
                 npix=128, 
                 oversample=2048/128,
                 npsf=400, 
                 psf_pixelscale=4.63e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None,
                 Imax_ref=None,
                 det_rotation=0,
                 source_offset=(0,0),
                 use_synthetic_opds=False,
                 use_measured_opds=False,
                 use_noise=False, # FIXME: Noise simulations must be implemented, at the very least implement shot noise
                 use_pupil_grating=False,
                 use_aps=False,
                 inf_fun=None,
                 dm_ref=np.zeros((34,34)),
                 bad_acts=None,
                 OPD=None,
                 RETRIEVED=None,
                 ZWFS=None,
                 FPM=None,
                 LYOT=None,
                 pupil_diam=6.75*u.mm,):
        
        
        print('ParallelizedScoob Initialized!')
        self.actors = actors
        self.Na = len(actors)

        # FIXME: Parameters that are in the model but needed higher up
        self.npix = ray.get(actors[0].getattr.remote('npix'))
        
        self.psf_pixelscale_lamD = ray.get(actors[0].getattr.remote('psf_pixelscale_lamD'))
        self.npsf = ray.get(actors[0].getattr.remote('npsf'))
        self.Imax_ref = Imax_ref
        
        self.Nact = 34
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        
        self.dm_mask = ray.get(actors[0].getattr.remote('dm_mask'))
    
    def calc_psfs(self, quiet=True):
        '''
        Calculate a psf for each wavelength.
        This wraps the calc_psf method in the SCOOBM class.
        Remember that this method returns a wavefront and not a psf.

        Returns
        -------

        psfs : `array`
            An array of the wavefronts at each wavelength.
        '''
        start = time.time()
        pending_psfs = []
        for i in range(len(self.actors)):
            future_psfs = self.actors[i].calc_psf.remote()
            pending_psfs.append(future_psfs)
        psfs = ray.get(pending_psfs)
        if isinstance(psfs[0], np.ndarray):
            xp = np
        elif isinstance(psfs[0], cp.ndarray):
            xp = cp
        psfs = xp.array(psfs)
        
        if not quiet: print('PSFs calculated in {:.3f}s.'.format(time.time()-start))
        return psfs
    
    def snaps(self):
        '''
        Generats a PSF for each actor (which is nominally each wavelength).
        This wraps the snap method in the SCOOBM class.

        Returns
        -------

        ims : `array`
            An array of PSFs with one slice per wavelength.
        '''
        pending_ims = []
        for i in range(len(self.actors)):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
            
        return ims

    def snap(self):
        '''
        Generates the PSF which is multiplied the desired flux 
        level for each wavelength and summed into a single image.

        FIXME: this is not yet implemented. 
        The goal is to have a single method that generates a polychromatic PSF.

        '''
        ims = snaps()

        # Creates a 2d array which will be the final PSF
        im=xp.zeros(ims.shape[1:3])

        # FIXME: this is better done by matrix multiplication and then
        # summing along an axis (im = xp.sum(ims, axis=0))
        for s in ims:
            im = im + (s * self.f_lambda[i])

        # flux_calibrate_psf(self, arr, f_lambda)
        
        return NotImplementedError()    

    def set_actor_attr(self, attr, value):
        '''
        Sets a value for all actors
        '''
        for i in range(len(self.actors)):
            self.actors[i].setattr.remote(attr,value)
    
    def set_dm(self, value):
        for i in range(len(self.actors)):
            self.actors[i].set_dm.remote(value)

    def add_dm(self, value):
        for i in range(len(self.actors)):
            self.actors[i].add_dm.remote(value)

    def get_dm(self):
        return ray.get(self.actors[0].get_dm.remote())
    
    def show_dm(self):
        wf = poppy.FresnelWavefront(beam_radius=self.dm_active_diam/2, npix=self.npix, oversample=1)
        dm_command = self.get_dm()
        dm_surface = self.DM.get_opd(wf).get() if poppy.accel_math._USE_CUPY else self.DM.get_opd(wf)
        surf_ext = wf.pixelscale.to_value(u.mm/u.pix)*self.npix/2
        
        imshows.imshow2(dm_command, dm_surface, 'DM Command', 'DM Surface')
        
    def calc_psf(self, quiet=True):
        '''
        Creates a the polychromatic wavefront at the focal plane which is scaled
        to the desired flux level for each wavelength.
        It is *not* normalized by the unocculted wavefront.
        
        This is the input for the EFC etc.

        Currently, there is no handling of normalization.
        '''

        # 
        psfs = self.calc_psfs(quiet=quiet)

        psf = self.flux_calibrate_wavefronts(psfs, self.f_lambda)
    
        return psf
    
    @classmethod
    def flux_calibrate_psf(self, arr, f_lambda, norm=None):
        '''
        Perform the flux calibration of a 3-Dimensional PSF (intensity) array 
        based on the desired spectral distributions.

        Parameters
        ----------
        
        arr : `cp.array`
            Array of psfs

        f_lambda : `list`
            Array of relative fluxes. This will be normalized such that
            the integral over the spectral range is 1.

        norm_arr : `cp.array`
            Optional array of occulted psfs. If provided, then the array will be
            divided by the maximum of this array. This is useful for putting
            occulted arrays of psfs into units of contrast.
        
        Returns
        -------

        im : `cp.array`
            Flux calibrated array which has been normalized by the maximum of 
            the maximum value of the norm array (if the norm parameter was 
            provided).
        
        f_lambda_norm : `np.array`
            Normalized flux array.

        '''

        # Set the norm array if it's not defined
        if norm is None:
            print('No normalization supplied.')
            norm=np.ones(len(arr))

        if len(arr) != len(norm):
            return IOError('length of input array is not of the same length as the normalization array')

        # Normalize the flux array
        f_lambda_norm = f_lambda/np.sum(f_lambda)

        f_im=xp.zeros((arr.shape[1:3]))
        f_norm=xp.zeros((arr.shape[1:3]))

        for i in range(len(arr)):
            f_im += (arr[i]/norm[i].max()) * f_lambda_norm[i]
            f_norm += (norm[i] * f_lambda_norm[i])

        return  f_im, f_norm

    @classmethod
    def flux_calibrate_wavefronts(self, arr, f_lambda):
        '''
        Perform the flux calibration of a 3-Dimensional array of wavefronts 
        based on the desired spectral distributions.

        Parameters
        ----------
        
        arr : `cp.array`
            Array of wavefronts

        f_lambda : `list`
            Array of relative fluxes. This will be normalized such that
            the integral over the spectral range is 1.
        
        Returns
        -------

        im : `cp.array`
            Flux calibrated wavefront.
        '''
        if len(arr) != len(f_lambda):
            return IOError(
                'length of input array is not of the same length as flux array'
                )

        # Normalize the flux array
        f_lambda_norm = f_lambda/xp.sum(f_lambda)
        # declare the output array
        f_wfr = xp.zeros((arr.shape[1:3]))
        
        for i in range(len(arr)):

            tmp= (arr[i] * f_lambda_norm[i])

            # Not sure why, but doing the following line results in a type error
            # f_wfr+=tmp
            f_wfr = f_wfr+tmp

        return  f_wfr