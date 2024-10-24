import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path

try:
    import ray
except ImportError:
    print('Unable to import ray; no parallelized propagation functionality available.')

from .math_module import xp,_scipy, ensure_np_array
from . import imshows
import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

class ParallelizedScoob():
    '''
    This is a class that sets up the parallelization of calc_psf such that it 
    we can generate polychromatic wavefronts that are then fed into the 
    various wavefront simulations.
    '''
    def __init__(self, 
                 actors,
                 ):
        
        self.actors = actors
        self.Nactors = len(actors)
        self.wavelength_c = self.getattr('wavelength_c')

        self.npix = ray.get(actors[0].getattr.remote('npix'))
        self.oversample = ray.get(actors[0].getattr.remote('oversample'))
        
        self.psf_pixelscale_lamD = ray.get(actors[0].getattr.remote('psf_pixelscale_lamD'))
        self.npsf = ray.get(actors[0].getattr.remote('npsf'))

        self.llowfsc_mode = ray.get(self.actors[0].getattr.remote('llowfsc_mode'))
        self.nllowfsc = ray.get(actors[0].getattr.remote('nllowfsc'))
        self.llowfsc_pixelscale = ray.get(actors[0].getattr.remote('llowfsc_pixelscale'))

        self.dm_mask = ray.get(actors[0].getattr.remote('dm_mask'))
        self.Nact = self.dm_mask.shape[0]
        self.dm_ref = ray.get(actors[0].getattr.remote('dm_ref'))

        self.Imax_ref = 1
        self.normalize = False
        self.use_noise = False

        self.optical_throughput = 0.1
        self.qe = 0.5
        self.read_noise = 1.5
        self.bias = 10
        self.gain = 1
        self.nbits = 16
        self.sat_thresh = 2**self.nbits - 1000
        self.subtract_bias = False
        
        self.exp_time = 1
        self.Nframes = 1

        self.exp_times = None
        self.Nframes_per_exp = None

    def getattr(self, attr):
        return ray.get(self.actors[0].getattr.remote(attr))
    
    def set_actor_attr(self, attr, value):
        '''
        Sets a value for all actors
        '''
        for i in range(len(self.actors)):
            self.actors[i].setattr.remote(attr,value)
    
    def reset_dm(self):
        self.set_dm(self.dm_ref)
    
    def zero_dm(self):
        self.set_dm(np.zeros((34,34)))
        
    def set_dm(self, value):
        for i in range(len(self.actors)):
            self.actors[i].set_dm.remote(value)

    def add_dm(self, value):
        for i in range(len(self.actors)):
            self.actors[i].add_dm.remote(value)

    def get_dm(self):
        return ray.get(self.actors[0].get_dm.remote())
    
    def use_scc(self, use=True):
        for i in range(len(self.actors)):
            self.actors[i].use_scc.remote(use)

    def use_llowfsc(self, use=True):
        for i in range(len(self.actors)):
            self.actors[i].use_llowfsc.remote(use)
        self.llowfsc_mode = ray.get(self.actors[0].getattr.remote('llowfsc_mode'))

    def block_lyot(self, val=True):
        for i in range(len(self.actors)):
            self.actors[i].block_lyot.remote(val)

    def add_noise(self, flux_image):
        # flux_image is provided as an array in units of photons/sec/pixel
        counts = flux_image * self.exp_time * self.optical_throughput * self.qe
        noisy_im = xp.random.poisson(counts) * self.gain # this step should take into account electrons per ADU
        noisy_im = noisy_im.astype(xp.float64) + self.bias
        noisy_im += int(xp.round(xp.random.normal(self.read_noise)))

        noisy_im[noisy_im>(2**self.nbits - 1)] = 2**self.nbits - 1
        if xp.max(noisy_im)==int(2**self.nbits - 1): print('WARNING: Image became saturated')
        return noisy_im

    def snap(self):
        if self.exp_times is not None and self.Nframes_per_exp is not None:
            return self.snap_many(plot=True)
            
        pending_ims = []
        for i in range(self.Nactors):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        im = xp.sum(ims, axis=0)

        if self.use_noise:
            mean_frame = 0.0
            for i in range(self.Nframes):
                mean_frame += self.add_noise(im)/self.Nframes
            im = mean_frame

            if self.normalize:
                im = im.astype(xp.float64) / self.exp_time / self.gain

        im = im.astype(xp.float64)/self.Imax_ref

        return im

    def snap_many(self, plot=False):
        pending_ims = []
        for i in range(self.Nactors):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        flux_im = xp.sum(ims, axis=0)

        print(self.exp_times, self.Nframes_per_exp)
        if len(self.exp_times)!=len(self.Nframes_per_exp):
            raise ValueError('The specified number of frames per exposure time must match the specified number of exposure times.')
            
        total_im = 0.0
        pixel_weights = 0.0
        for i in range(len(self.exp_times)):
            self.exp_time = self.exp_times[i]
            self.Nframes = self.Nframes_per_exp[i]
            mean_frame = 0.0
            for j in range(self.Nframes):
                mean_frame += self.add_noise(flux_im)/self.Nframes
                
            pixel_sat_mask = mean_frame > self.sat_thresh

            if self.subtract_bias is not None:
                mean_frame -= self.subtract_bias
            
            pixel_weights += ~pixel_sat_mask
            normalized_im = mean_frame/self.exp_time 
            normalized_im[pixel_sat_mask] = 0 # mask out the saturated pixels

            if plot: 
                imshows.imshow3(pixel_weights, mean_frame, normalized_im, 
                                'Pixel Weight Map', 
                                f'Frame:\nExposure Time = {self.exp_time:.2e}s', 
                                'Masked Flux Image', 
                                # lognorm2=True, lognorm3=True,
                                )
                
            total_im += normalized_im
            
        total_im /= pixel_weights

        return total_im

    
        
        
    