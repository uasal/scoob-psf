from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows
from . import dm
from . import props

import poppy

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

class CORO():

    def __init__(self, 
                 wavelength=None, 
                 npsf=200,
                 psf_pixelscale=3.76e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 dm_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 Imax_ref=1,
                 use_scc=False, 
                 ):
        
        self.wavelength_c = 633e-9*u.m
        self.pupil_diam = 6.75*u.mm
        self.dm_pupil_diam = 9.4*u.mm
        self.lyot_pupil_diam = 9.4*u.mm
        self.lyot_diam = 8.6*u.mm

        self.lyot_ratio = 8.6/9.4

        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.npix = 500
        self.oversample = 4.096
        self.npix = 1000
        self.oversample = 2.5
        self.N = int(self.npix*self.oversample)
        self.Nfpm = 4096

        self.imaging_fl = 300*u.mm

        self.use_fpm = False
        self.use_scc = use_scc

        pwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=1)
        self.APERTURE = poppy.CircularAperture(radius=self.pupil_diam/2).get_transmission(pwf)
        self.APMASK = self.APERTURE>0
        if self.use_scc:
            lwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=self.oversample)
            lyot = poppy.CircularAperture(radius=self.lyot_ratio*self.pupil_diam/2).get_transmission(lwf)
            scc_pinhole = poppy.CircularAperture(radius=0.03*self.lyot_ratio*self.pupil_diam/2, 
                                                 shift_x=1.55/np.sqrt(2)*self.pupil_diam, shift_y=1.55/np.sqrt(2)*self.pupil_diam).get_transmission(lwf)
            self.LYOT = lyot + scc_pinhole
        else:
            self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.pupil_diam/2).get_transmission(pwf)
            
        self.WFE = xp.ones((self.npix,self.npix), dtype=complex)

        self.um_per_lamD = (self.wavelength_c*self.imaging_fl/(self.lyot_diam)).to(u.um)

        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = self.psf_pixelscale.to_value(u.um/u.pix)/self.um_per_lamD.value
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = self.psf_pixelscale_lamD * self.um_per_lamD/u.pix

        self.dm_ref = dm_ref
        self.init_dm()
        self.reset_dm()

        self.Imax_ref = Imax_ref
        self.reverse_parity = False

    def getattr(self, attr):
        return getattr(self, attr)
    
    @property
    def psf_pixelscale(self):
        return self._psf_pixelscale
    
    @psf_pixelscale.setter
    def psf_pixelscale(self, value):
        self._psf_pixelscale = value.to(u.m/u.pix)
        self.psf_pixelscale_lamD = (self._psf_pixelscale / self.um_per_lamD).decompose().value

    def init_dm(self):
        act_spacing = 300e-6*u.m
        pupil_pxscl = self.dm_pupil_diam.to_value(u.m)/self.npix
        sampling = act_spacing.to_value(u.m)/pupil_pxscl
        print('influence function sampling', sampling)
        inf, inf_sampling = dm.make_gaussian_inf_fun(act_spacing=act_spacing, sampling=sampling, coupling=0.15,)
        self.DM = dm.DeformableMirror(inf_fun=inf, inf_sampling=sampling, name='DM')

        self.Nact = self.DM.Nact
        self.Nacts = self.DM.Nacts
        self.act_spacing = self.DM.act_spacing
        self.dm_active_diam = self.DM.active_diam
        self.dm_full_diam = self.DM.pupil_diam
        
        self.full_stroke = self.DM.full_stroke
        
        self.dm_mask = self.DM.dm_mask

    def reset_dm(self):
        self.set_dm(self.dm_ref)

    def zero_dm(self):
        self.set_dm(xp.zeros((self.Nact,self.Nact)))
    
    def set_dm(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM.command = dm_command
        
    def add_dm(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM.command += dm_command
        
    def get_dm(self):
        return self.DM.command
    
    def map_actuators_to_command(self, act_vector):
        command = np.zeros((self.Nact, self.Nact))
        command.ravel()[self.dm_mask.ravel()] = ensure_np_array(act_vector)
        return command
    
    def calc_wfs(self, save_wfs=True, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        wfs = []
        self.wf = utils.pad_or_crop(self.APERTURE, self.N).astype(complex)
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf *= utils.pad_or_crop(self.WFE, self.N)
        if save_wfs: wfs.append(copy.copy(self.wf))

        dm_surf = utils.pad_or_crop(self.DM.get_surface(), self.N)
        self.wf *= xp.exp(1j*4*np.pi*dm_surf/self.wavelength.to_value(u.m))
        if save_wfs: wfs.append(copy.copy(self.wf))

        if self.use_fpm:
            self.wf = props.apply_vortex(utils.pad_or_crop(self.wf, self.Nfpm), npix=self.npix)
            self.wf = utils.pad_or_crop(self.wf, self.N)
            # self.wf = props.apply_vortex(utils.pad_or_crop(self.wf, self.npix), Nfpm=self.Nfpm, N=self.N, plot=False)
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf *= utils.pad_or_crop(self.LYOT, self.N).astype(complex)
        if self.reverse_parity: self.wf = xp.rot90(xp.rot90(self.wf))
        if save_wfs: wfs.append(copy.copy(self.wf))

        # Nlyot = int(np.round(self.lyot_ratio * self.npix))
        # self.wf = props.mft_forward(utils.pad_or_crop(self.wf, Nlyot), self.psf_pixelscale_lamD, self.npsf)
        self.wf = props.mft_forward(self.wf, self.psf_pixelscale_lamD * self.oversample / self.lyot_ratio, self.npsf)
        self.wf /= xp.sqrt(self.Imax_ref) # normalize by a reference maximum value
        if save_wfs: wfs.append(copy.copy(self.wf))

        if save_wfs:
            return wfs
        else:
            return self.wf
    
    def calc_wf(self):
        fpwf = self.calc_wfs(save_wfs=False, quiet=True)
        return fpwf
    
    def snap(self):
        image = xp.abs(self.calc_wfs(save_wfs=False, quiet=True))**2
        return image
    


