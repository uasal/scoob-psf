import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import PchipInterpolator
import time
import os
from pathlib import Path
import copy

from .math_module import xp, _scipy, ensure_np_array
from .imshows import *
from . import utils, dm

escpsf_dir = os.path.dirname(__file__)

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image


class noisy_detector():

    def __init__(self, npix, exposure_time):
        
        # setup detector based on parameters
        self.quantum_efficiency = 0.9
        self.dark_current = 2.2e-3
        self.read_noise = 1.4
        self.exposure_time = exposure_time
        self.bit_depth = 2 ** 16

        # image parameters
        self.npix = npix
        
        # initialize 
        self.accumulated_charge = xp.zeros((self.npix, self.npix))

    def integrate(self, flux, exposure_time=None):
        if exposure_time is not None:

            # add incoming wavefront flux
            self.accumulated_charge += flux * exposure_time * self.quantum_efficiency

            # add dark current               
            self.accumulated_charge += self.dark_current * exposure_time * self.quantum_efficiency

        elif self.exposure_time is not None:
            # add incoming wavefront flux
            self.accumulated_charge += flux * self.exposure_time * self.quantum_efficiency

            # add dark current               
            self.accumulated_charge += self.dark_current * self.exposure_time * self.quantum_efficiency
        else:
            print("Please provide the detector an exposure time!")
    

    def read_out(self):

        # don't overwrite output
        image = self.accumulated_charge.copy()     

        # add photon noise                             
        image = large_poisson(image)                                   

        # add read noise         
        image += xp.random.normal(loc=0, size=xp.shape(image), scale=self.read_noise)  

        # round to an integer bit value
        image = xp.round(image)

        # anything above bit depth saturates
        image[image > self.bit_depth] = self.bit_depth

        # can't have negative counts
        image[image<0] = 0

        # reset charge
        self.accumulated_charge = xp.zeros((self.npix, self.npix))                                                  
            
        return image
    
def large_poisson(lam, thresh=1e6):
    '''
    Taken from HCIPy with a few tweaks for cupy integration.

	Draw samples from a Poisson distribution, taking care of large values of `lam`.

	At large values of `lam` the distribution automatically switches to the corresponding normal distribution.
	This switch is independently decided for each expectation value in the `lam` array.

	Parameters
	----------
	lam : array_like
		Expectation value for the Poisson distribution. Must be >= 0.
	thresh : float
		The threshold at which the distribution switched from a Poisson to a normal distribution.

	Returns
	-------
	array_like
		The drawn samples from the Poisson or normal distribution, depending on the expectation value.
	'''
    # save original shape then flatten input
    orig_shape = lam.shape
    lam = lam.ravel()

    # where are the large values
    large = lam > thresh

    # if they ain't large, then they smol
    small = ~large

    # use a normal approximation if the number of photons is large
    n = xp.zeros(lam.shape)
    n[large] = xp.round(lam[large] + xp.random.normal(size=int(xp.sum(large))) * xp.sqrt(lam[large]))

    # aaaaand use an actual poisson distribution if the number of photons is smol
    n[small] = xp.random.poisson(lam[small], size=int(xp.sum(small)))

    return n.reshape(orig_shape)