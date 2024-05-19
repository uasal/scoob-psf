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

    def snap(self):
        pending_ims = []
        for i in range(self.Nactors):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        im = xp.sum(ims, axis=0)

        return im
        
        
    