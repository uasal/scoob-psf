
   
'''
Basic implementation of an Ideal AGPM
following the POPPY nomenclature (v0.5.1)
@author Gilles Orban de Xivry (ULg)
@date 12 / 02 / 2017
'''
from __future__ import division
import poppy
from poppy.poppy_core import Wavefront, PlaneType
from poppy.fresnel import FresnelWavefront
from poppy import utils

import numpy as np
import astropy.units as u

from .math_module import xp,_scipy

class IdealAGPM(poppy.AnalyticOpticalElement):
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
    def __init__(self, name="unnamed AGPM ",
                 wavelength=1e-6 * u.meter,
                 charge=2,
                 singularity=None,
                 centering='PIXEL', # PIXEL or INTERPIXEL
                 **kwargs):
        
        poppy.AnalyticOpticalElement.__init__(self, planetype=PlaneType.intermediate, **kwargs)
        self.name = name
        self.lp = charge
        self.singularity = singularity
        self.central_wavelength = wavelength

        self.centering = centering
        
    def get_phasor(self, wave):
        """
        Compute the amplitude transmission appropriate for a vortex for
        some given pixel spacing corresponding to the supplied Wavefront
        """

#         if not isinstance(wave, Wavefront) and not isinstance(wave, FresnelWavefront):  # pragma: no cover
#             raise ValueError("AGPM get_phasor must be called with a Wavefront "
#                              "to define the spacing")
#         assert (wave.planetype != PlaneType.image)

        y, x = self.get_coordinates(wave)
        phase = xp.arctan2(y, x)

        AGPM_phasor = xp.exp(1.j * self.lp * phase) * self.get_transmission(wave)
        
#         print(x)
        idx = xp.where(x==0)[0][0]
        idy = xp.where(y==0)[0][0]
        AGPM_phasor[idx, idy] = 0
        return AGPM_phasor

    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)
        phase = xp.arctan2(y, x)
        return self.lp * phase * self.central_wavelength.to(u.meter).value / (2 * np.pi)

    def get_transmission(self, wave):
        y, x = self.get_coordinates(wave)
        
        if self.singularity is None:
            trans = xp.ones(y.shape)
        else:
            circ = poppy.InverseTransmission(poppy.CircularAperture(radius=self.singularity/2))
            trans = circ.get_transmission(wave)
            
        return trans
    
    