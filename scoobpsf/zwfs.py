import poppy
from poppy.poppy_core import Wavefront, PlaneType
from poppy.fresnel import FresnelWavefront
from poppy import utils

import numpy as np
import astropy.units as u

from .math_module import xp,_scipy

class IdealZWFS(poppy.AnalyticOpticalElement):
    """ Defines an ideal vortex phase mask coronagraph.
    Parameters
    ----------
    name : string
        Descriptive name
    wavelength : float
        Wavelength in meters.
    charge : int
        Charge of the vortex
    """
    @utils.quantity_input(wavelength=u.meter)
    def __init__(self, spatial_resolution,
                 name="unnamed ZWFS",
                 central_wavelength=1e-6 * u.meter,
                 diameter=1.06, # in lam/D
                 **kwargs):
        
        poppy.AnalyticOpticalElement.__init__(self, planetype=PlaneType.intermediate, **kwargs)
        self.name = name
        self.spatial_resolution = spatial_resolution
        self.central_wavelength = central_wavelength
        self.diameter = diameter
        
    def get_phasor(self, wave):
        """
        Compute the amplitude transmission appropriate for a vortex for
        some given pixel spacing corresponding to the supplied Wavefront
        """
        
        y, x = self.get_coordinates(wave)
        r = xp.sqrt(x**2 + y**2)
        
        phase = np.pi/2 * xp.ones_like(r)
        phase *= r<self.diameter*self.spatial_resolution.to_value(u.m)/2
        
        zwfs_phasor = xp.exp(1j*phase)
        
        return zwfs_phasor

    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)
        r = xp.sqrt(x**2 + y**2)
        
        phase = np.pi/2 * xp.ones_like(r)
        return phase * self.central_wavelength.to(u.meter).value / (2 * np.pi)

    def get_transmission(self, wave):
        trans = xp.ones_like(wave.wavefront)
        return trans
    