from .math_module import xp,_scipy, ensure_np_array
import scoobpsf.utils as utils

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
from pathlib import Path

import poppy

def make_gaussian_inf_fun(act_spacing=300e-6*u.m, sampling=10, coupling=0.15, Nact=4):
    ng = int(sampling*Nact)
    pxscl = act_spacing/(sampling*u.pix)

    xs = (xp.linspace(-ng/2,ng/2-1,ng)+1/2)*pxscl.to_value(u.m/u.pix)
    x,y = xp.meshgrid(xs,xs)
    r = xp.sqrt(x**2 + y**2)

    d = act_spacing.to_value(u.m)/np.sqrt(-np.log(coupling))

    inf_fun = np.exp(-(r/d)**2)

    return inf_fun

class DeformableMirror(poppy.AnalyticOpticalElement):
    
    def __init__(self,
                 inf_fun,
                 inf_sampling,
                 Nact=34,
                 act_spacing=300e-6*u.m,
                 max_stroke=1500e-9, 
                 Nbits=16, 
                 aperture=None,
                 include_reflection=True,
                 planetype=poppy.poppy_core.PlaneType.intermediate,
                 name='DM',
                ):
        
        self.inf_fun = inf_fun
        self.inf_sampling = inf_sampling

        self.Nact = Nact
        self.act_spacing = act_spacing
        self.include_reflection = include_reflection

        self.max_stroke = max_stroke
        self.Nbits = Nbits
        self.Nvals = 2**Nbits
        self.avail_act_vals = xp.linspace(-self.max_stroke/2, self.max_stroke/2, self.Nvals, dtype=xp.float64)
        self.act_res = self.avail_act_vals[1] - self.avail_act_vals[0]
        self.use_act_res = False

        self.Nsurf = inf_fun.shape[0]
        self.pixelscale = self.act_spacing/(self.inf_sampling*u.pix)
        self.active_diam = self.Nact * self.act_spacing

        self.yc, self.xc = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
        self.rc = xp.sqrt(self.xc**2 + self.yc**2)
        self.dm_mask = self.rc<(Nact/2 + 1/2)
        self.Nacts = int(xp.sum(self.dm_mask))
        
        self.command = xp.zeros((self.Nact, self.Nact))
        self.actuators = xp.zeros(self.Nacts)

        self.aperture = aperture
        self.planetype = planetype
        self.name = name

        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))
        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        x = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)
        y = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,x))
        self.My = xp.exp(-1j*2*np.pi*xp.outer(y,fy))

        self.pxscl_tol = 1e-6

    def quantize_acts(self, acts):
        quantized_acts = xp.zeros(self.Nacts)
        for i in range(self.Nacts):
            nearest_act_ind = xp.argmin(xp.abs(acts[i] - self.avail_act_vals))
            nearest_act_val = self.avail_act_vals[nearest_act_ind]
            # print(nearest_act_ind, nearest_act_val)
            quantized_acts[i] = nearest_act_val 
        return quantized_acts

    @property
    def command(self):
        return self._command

    @command.setter
    def command(self, command_values):
        command_values *= self.dm_mask
        if self.use_act_res: 
            acts = command_values[self.dm_mask]
            quantized_acts = self.quantize_acts(acts)
            command_values = xp.zeros_like(command_values)
            command_values[self.dm_mask] = quantized_acts
        self._actuators = self.map_command_to_actuators(command_values) # ensure you update the actuators if command is set
        self._command = command_values
    
    @property
    def actuators(self):
        return self._actuators

    @actuators.setter
    def actuators(self, act_vector):
        if self.use_act_res: 
            act_vector = self.quantize_acts(act_vector)
        self._command = self.map_actuators_to_command(act_vector) # ensure you update the actuators if command is set
        self._actuators = act_vector
    
    def map_command_to_actuators(self, command_values):
        actuators = command_values.ravel()[self.dm_mask.ravel()]
        return actuators
        
    def map_actuators_to_command(self, act_vector):
        command = xp.zeros((self.Nact, self.Nact))
        command[self.dm_mask] = act_vector
        return command
    
    def get_surface(self):
        mft_command = self.Mx@self.command@self.My
        fourier_surf = self.inf_fun_fft * mft_command
        surf = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(fourier_surf,))).real
        return surf
    
    # METHODS TO BE COMPATABLE WITH POPPY
    def get_opd(self, wave):
        opd = self.get_surface()
        if self.include_reflection:
            opd *= 2

        pxscl_diff = wave.pixelscale.to_value(u.m/u.pix) - self.pixelscale.to_value(u.m/u.pix) 
        if pxscl_diff < self.pxscl_tol:
            opd = utils.interp_arr(opd, self.pixelscale.to_value(u.m/u.pix), wave.pixelscale.to_value(u.m/u.pix) )
        
        opd = utils.pad_or_crop(opd, wave.shape[0])

        return opd

    def get_transmission(self, wave):
        if self.aperture is None:
            trans = xp.ones_like(wave.wavefront)
        else:
            trans = self.aperture.get_transmission(wave)
        return trans
    
    def get_phasor(self, wave):
        """
        Compute the amplitude transmission appropriate for a vortex for
        some given pixel spacing corresponding to the supplied Wavefront
        """

        assert (wave.planetype != poppy.poppy_core.PlaneType.image)

        dm_phasor = self.get_transmission(wave) * xp.exp(1j * 2*np.pi/wave.wavelength.to_value(u.m) * self.get_opd(wave))

        return dm_phasor
        

