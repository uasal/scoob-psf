import numpy as np
# import cupy as cp
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

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image


class ParallelizedScoob():
    '''
    This is a class that sets up the parallelization of calc_psf such that it 
    we can generate polychromatic wavefronts that are then fed into the 
    various wavefront simulations.
    '''
    def __init__(self, 
                 actors,
                 dm_ref=np.zeros((34,34)),
                 use_noise=False,
                 exp_time=None,
                 Imax_ref=None,
                 normalize=False):
        
        
        print('ParallelizedScoob Initialized!')
        self.actors = actors
        self.Na = len(actors)

        # FIXME: Parameters that are in the model but needed higher up
        self.npix = ray.get(actors[0].getattr.remote('npix'))
        self.oversample = ray.get(actors[0].getattr.remote('oversample'))
        
        self.psf_pixelscale_lamD = ray.get(actors[0].getattr.remote('psf_pixelscale_lamD'))
        self.det_rotation = ray.get(actors[0].getattr.remote('det_rotation'))
        self.npsf = ray.get(actors[0].getattr.remote('npsf'))
        
        self.use_noise = use_noise
        self.exp_time = exp_time
        self.Imax_ref = Imax_ref
        self.normalize = normalize
        
        self.Nact = 34
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        
        self.dm_mask = ray.get(actors[0].getattr.remote('dm_mask'))
        self.dm_ref = dm_ref
        self.set_dm(dm_ref)

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
    
    def show_dm(self):
        imshows.imshow1(self.get_dm(), 'DM Command',)
        
    def calc_psfs(self, quiet=True):
        start = time.time()
        pending_psfs = []
        for i in range(self.Na):
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
    
    def snap(self):
        pending_ims = []
        for i in range(self.Na):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        im = xp.sum(ims, axis=0)/self.Na
        
        if self.normalize:
            if self.Imax_ref is not None:
                im /= self.Imax_ref
                
            if self.exp_time is not None and self.exp_time_ref is not None:
                im /= (self.exp_time/self.exp_time_ref).value
            
            
        return im
        
        
        
    