import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import pickle
import time
import copy

import poppy

from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image

try:
    import cupy as cp
    import cupyx.scipy as _scipy
    xp = cp
except ImportError:
    cp = None
    xp = np
    _scipy = scipy

def ensure_np_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return arr.get()
    
import misc_funs as misc

class DeformableMirror():
    
    
    def __init__(self, 
                 Nact=34,
                 act_spacing=300e-6*u.m,
                 pupil_diam=11.1*u.mm,
                 full_stroke=1.5e-6*u.m,
                 inf_fun=None,
                 inf_sampling=None,
                ):
        
        self.Nact = Nact
        self.act_spacing = act_spacing
        self.active_diam = self.Nact * self.act_spacing
        self.pupil_diam = pupil_diam
        
        self.full_stroke = full_stroke
        
        self.dm_mask = xp.ones((self.Nact,self.Nact), dtype=bool)
        xx = (xp.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * 2*self.act_spacing.to_value(u.m)
        x,y = xp.meshgrid(xx,xx)
        r = xp.sqrt(x**2 + y**2)
        self.dm_mask[r>(self.active_diam + self.act_spacing).to_value(u.m)] = 0
        
        self.Nacts = int(np.sum(ensure_np_array(self.dm_mask)))
        
        self.command = xp.zeros((self.Nact, self.Nact))
        self.actuators = xp.zeros(self.Nacts)
        
        if inf_fun is None:
            inf_fun = xp.array(fits.getdata('inf.fits'))
            inf_sampling = fits.getheader('inf.fits')['SAMPLING']
            
        self.inf_fun = inf_fun
        self.inf_sampling = inf_sampling # number of pixels per actuator
        self.Nact_per_inf = (inf_fun.shape[0]-1)/inf_sampling # number of actuators across one influence function grid
        self.inf_pixelscale = self.act_spacing/(self.inf_sampling*u.pix)
        
    @property
    def command(self):
        return self._command

    @command.setter
    def command(self, command_values):
        command_values *= self.dm_mask
        self._actuators = self.map_command_to_actuators(command_values) # ensure you update the actuators if command is set
        self._command = command_values
    
    @property
    def actuators(self):
        return self._actuators

    @actuators.setter
    def actuators(self, act_vector):
        self._command = self.map_actuators_to_command(act_vector) # ensure you update the actuators if command is set
        self._actuators = act_vector
        
    def map_command_to_actuators(self, command_values):
        actuators = command_values.ravel()[self.dm_mask.ravel()]
        return actuators
        
    def map_actuators_to_command(self, act_vector):
        command = xp.zeros((self.Nact, self.Nact))
        command.ravel()[self.dm_mask.ravel()] = act_vector
        return command
    
    def get_surface(self, npix):
        Ninf = self.inf_fun.shape[0]
        npact = self.inf_sampling
        nact = self.Nact_per_inf
        
        Nsurf = (self.Nact-1)*npact+npact+1
        Nover = int(Nsurf+2*np.floor(nact/2)*npact)
        oversized_surface = xp.zeros((Nover,Nover))
        
        for j in range(self.Nact):
            for i in range(self.Nact):
                DMi = xp.zeros_like(oversized_surface)
                DMi[npact*(i):Ninf+npact*(i),npact*(j):Ninf+npact*(j)] = self.command[i,j]*self.inf_fun
                oversized_surface += DMi
        
        surf = oversized_surface[np.floor(nact/2)*npact:-np.floor(nact/2)*npact,
                                 np.floor(nact/2)*npact:-np.floor(nact/2)*npact]
        
        new_pixelscale = self.active_diam/(npix*u.pix)
        interped_surf = interp_2d_array(surf, self.inf_pixelscale.to_value(u.m/u.pix), new_pixelscale.to_value(u.m/u.pix))
        return interped_surf


def interp_2d_array(arr, pixelscale, new_pixelscale):
    Nold = arr.shape[0]
    old_xmax = pixelscale * Nold/2

    x,y = xp.ogrid[-old_xmax:old_xmax-pixelscale:Nold*1j,
                   -old_xmax:old_xmax-pixelscale:Nold*1j]

    Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) + 1
    new_xmax = new_pixelscale * Nnew/2

    newx,newy = xp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                         -new_xmax:new_xmax-new_pixelscale:Nnew*1j]

    x0 = x[0,0]
    y0 = y[0,0]
    dx = x[1,0] - x0
    dy = y[0,1] - y0

    ivals = (newx - x0)/dx
    jvals = (newy - y0)/dy

    coords = xp.array([ivals, jvals])

    interped_arr = _scipy.ndimage.map_coordinates(arr, coords, order=3)
    return interped_arr



