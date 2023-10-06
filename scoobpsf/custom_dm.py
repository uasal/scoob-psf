import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import pickle
import time
import copy
import os

import scoobpsf
from scoobpsf import imshows
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

from .math_module import xp,_scipy, ensure_np_array
from . import imshows
from . utils import pad_or_crop

import matplotlib.patches as patches

def make_gaussian_inf_fun(act_spacing=300e-6*u.m, sampling=25, coupling=0.15,
                            plot=False,
                            save_fits=None):

    Nacts_per_inf = 4 # number of influence functions across the grid
    ng = int(sampling*Nacts_per_inf) + 1

    pxscl = act_spacing/(sampling*u.pix)
    ext = Nacts_per_inf * act_spacing

    xs = (np.linspace(-ng/2,ng/2-1,ng)+1/2)*pxscl.value
    x,y = np.meshgrid(xs,xs)
    r = np.sqrt(x**2 + y**2)

    # d = act_spacing.value/1.25
    d = act_spacing.value/np.sqrt(-np.log(coupling))
    print(d)

    inf = np.exp(-(r/d)**2)
    rcoupled = d*np.sqrt(-np.log(coupling)) 

    if plot:
        fig,ax = imshow1(inf, pxscl=pxscl, patches=[patches.Circle((0,0), rcoupled, fill=False, color='c')], 
                            display_fig=False, return_fig=True)
        ticks = np.linspace(-ext.value/2, ext.value/2, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid()
        display(fig)

    if save_fits is not None:
        hdr = fits.Header()
        hdr['SAMPLING'] = sampling
        hdr.comments['SAMPLING'] = '# pixels per actuator'
        hdr['NACTS'] = ext_act
        hdr.comments['NACTS'] = '# actuators across grid'
        inf_hdu = fits.PrimaryHDU(data=inf, header=hdr)
        inf_hdu.writeto(str(save_fits), overwrite=True)

    return inf, sampling

class DeformableMirror():
    
    
    def __init__(self, 
                 Nact=34,
                 act_spacing=300e-6*u.m,
                 pupil_diam=11.1*u.mm,
                 full_stroke=1.5e-6*u.m,
                 inf_fun=None,
                 inf_cube=None,
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
        self.Nacts = int(xp.sum(self.dm_mask))
        
        self.command = xp.zeros((self.Nact, self.Nact))
        self.actuators = xp.zeros(self.Nacts)
            
        if isinstance(inf_cube, str):    
            self.inf_sampling = fits.getheader(inf_cube)['SAMPLING']
            self.inf_cube = xp.array(fits.getdata(inf_cube))
        elif isinstance(inf_cube, np.ndarray) or isinstance(inf_cube, xp.ndarray):
            if inf_sampling is None:
                raise ValueError('Must supply influence function sampling if providing a numerical array')
            self.inf_cube = xp.array(inf_cube)
            self.inf_sampling = inf_sampling
        elif inf_cube is None:
            if inf_fun is None:
                inf_fpath = str(module_path/'inf.fits')
                self.inf_fun = xp.array(fits.getdata(inf_fpath))
                self.inf_sampling = fits.getheader(inf_fpath)['SAMPLING']
            elif isinstance(inf_fun, str):
                self.inf_fun = xp.array(fits.getdata(inf_fun))
                self.inf_sampling = fits.getheader(inf_fun)['SAMPLING']
            elif isinstance(inf_fun, xp.ndarray):
                if inf_sampling is None:
                    raise ValueError('Must supply influence function sampling if providing a numerical array')
                self.inf_fun = inf_fun
                self.inf_sampling = inf_sampling
            self.build_inf_cube()
        self.inf_matrix = self.inf_cube.reshape(self.Nacts, self.inf_cube.shape[1]**2,).T
        
#         self.Nact_per_inf = (self.inf_fun.shape[0]-1)/self.inf_sampling # number of actuators across one influence function grid
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
        command[self.dm_mask] = act_vector
        return command
    
    def build_inf_cube(self, overpad=None):
        
        Nsurf = int(self.inf_sampling * self.Nact)
        if overpad is None:
            overpad = round(self.inf_sampling)
        
        Nsurf_padded = Nsurf + overpad
        
        padded_inf_fun = pad_or_crop(self.inf_fun, Nsurf_padded)

        self.inf_cube = xp.zeros((self.Nacts, Nsurf_padded, Nsurf_padded))
        act_inds = xp.argwhere(self.dm_mask)

        for i in range(self.Nacts):
            Nx = int(xp.round((act_inds[i][1] + 1/2 - self.Nact/2) * self.inf_sampling))
            Ny = int(xp.round((act_inds[i][0] + 1/2 - self.Nact/2) * self.inf_sampling))
            shifted_inf_fun = _scipy.ndimage.shift(padded_inf_fun, (Ny,Nx))
            self.inf_cube[i] = shifted_inf_fun
        
        return
        
    def get_surface(self, pixelscale=None):
        
        surf = self.inf_matrix.dot(self.actuators).reshape(self.inf_cube.shape[1], self.inf_cube.shape[1])
        
        if pixelscale is None:
            return surf
        else:
            surf = self.interp_surf(surf, self.inf_pixelscale.to_value(u.m/u.pix), pixelscale.to_value(u.m/u.pix))
            return surf
    
    def interp_surf(self, surf, pixelscale, new_pixelscale):
        Nold = surf.shape[0]
        old_xmax = pixelscale * Nold/2

        x,y = xp.ogrid[-old_xmax:old_xmax-pixelscale:Nold*1j,
                       -old_xmax:old_xmax-pixelscale:Nold*1j]

        Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) - 1
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

        interped_arr = _scipy.ndimage.map_coordinates(surf, coords, order=3)
        return interped_arr
        



