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
                 entrance_flux=None, 
                 use_fieldstop=False,
                 scc_diam=None,
                 scc_pinhole_position=None, 
                 ):
        
        self.wavelength_c = 633e-9*u.m
        self.pupil_diam = 6.75*u.mm
        self.dm_pupil_diam = 9.4*u.mm
        self.lyot_pupil_diam = 9.4*u.mm
        self.lyot_diam = 8.6*u.mm

        self.total_pupil_diam = 2.4*u.m # the assumed total pupil size for the actual telescope

        self.lyot_ratio = 8.6/9.4

        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.use_fpm = False
        self.use_fieldstop = use_fieldstop

        self.return_pupil = False
        self.scc_mode = False

        self.scc_diam = scc_diam
        if scc_pinhole_position is None:
            shift_x = 1.55/np.sqrt(2)*self.lyot_diam
            shift_y = 1.55/np.sqrt(2)*self.lyot_diam
        else:
            shift_x = scc_pinhole_position[0]
            shift_y = scc_pinhole_position[1]
        self.scc_pinhole_position = [shift_x, shift_y]

        self.llowfsc_mode = False
        self.llowfsc_pixelscale = 3.76*u.um/u.pix
        self.llowfsc_defocus = 1.75*u.mm
        self.llowfsc_fl = 200*u.mm
        self.nllowfsc = 64
        
        self.npix = 1000
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)
        self.Nfpm = 4096

        self.imaging_fl = 300*u.mm

        pwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=1) # pupil wavefront
        self.APERTURE = poppy.CircularAperture(radius=self.pupil_diam/2).get_transmission(pwf)
        self.APMASK = self.APERTURE>0
        self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.pupil_diam/2).get_transmission(pwf)

        if self.use_fieldstop:
            fp_pixelscale = 1/self.oversample
            x = xp.linspace(-self.N//2, self.N//2-1, self.N) * fp_pixelscale * self.lyot_ratio
            x,y = xp.meshgrid(x,x)
            r = xp.sqrt(x**2 + y**2)
            self.FIELDSTOP = r<=10
            
        self.WFE = xp.ones((self.npix,self.npix), dtype=complex)

        self.um_per_lamD = (self.wavelength*self.imaging_fl/(self.lyot_diam)).to(u.um)

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
        self.entrance_flux = entrance_flux
        if self.entrance_flux is not None:
            pixel_area = (self.total_pupil_diam/self.npix)**2
            flux_per_pixel = self.entrance_flux * pixel_area
            self.APERTURE *= xp.sqrt(flux_per_pixel.to_value(u.photon/u.second))

        self.reverse_parity = False

        self.use_oap_ap = False

    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

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
        # print('influence function sampling', sampling)
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
    
    def use_scc(self, use=True):
        if use: 
            self.scc_mode = True
            self.oversample = 2.5
            self.N = int(self.npix*self.oversample)

            lwf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, npix=self.npix, oversample=self.oversample)
            lyot = poppy.CircularAperture(radius=self.lyot_diam/2).get_transmission(lwf)
            scc_pinhole = poppy.CircularAperture(radius=self.scc_diam/2, 
                                                shift_x=self.scc_pinhole_position[0], 
                                                shift_y=self.scc_pinhole_position[1]).get_transmission(lwf)
            self.LYOT = lyot + scc_pinhole
        else:
            self.scc_mode = False
            self.oversample = 2.048
            self.N = int(self.npix*self.oversample)
            pwf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, npix=self.npix, oversample=1) # pupil wavefront
            self.LYOT = poppy.CircularAperture(radius=self.lyot_diam/2).get_transmission(pwf)

    def block_lyot(self, val=True):
        # only meant to be used if currently using an SCC as well
        self.llowfsc_mode = False
        if val:
            self.oversample = 2.5
            self.N = int(self.npix*self.oversample)
            lwf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, npix=self.npix, oversample=self.oversample)
            scc_pinhole = poppy.CircularAperture(radius=self.scc_diam/2, 
                                                 shift_x=self.scc_pinhole_position[0], 
                                                 shift_y=self.scc_pinhole_position[1]).get_transmission(lwf)
            self.LYOT = scc_pinhole
        else:
            self.oversample = 2.5
            self.N = int(self.npix*self.oversample)
            lwf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, npix=self.npix, oversample=self.oversample)
            lyot = poppy.CircularAperture(radius=self.lyot_diam/2).get_transmission(lwf)
            scc_pinhole = poppy.CircularAperture(radius=self.scc_diam/2, 
                                                 shift_x=self.scc_pinhole_position[0], 
                                                 shift_y=self.scc_pinhole_position[1]).get_transmission(lwf)
            lyot += scc_pinhole
            self.LYOT = lyot

    def use_llowfsc(self, use=True):
        if use:
            self.llowfsc_mode = True
            self.oversample = 4.096
            self.N = int(self.npix*self.oversample)
            lwf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, npix=self.npix, oversample=self.oversample)
            lyot = poppy.CircularAperture(radius=self.lyot_diam/2).get_transmission(lwf)
            if self.scc_diam is not None:
                scc_pinhole = poppy.CircularAperture(radius=self.scc_diam/2, 
                                                    shift_x=self.scc_pinhole_position[0], 
                                                    shift_y=self.scc_pinhole_position[1]).get_transmission(lwf)
                lyot += scc_pinhole
            self.LYOT = 1 - lyot
            self.LYOT *= poppy.CircularAperture(radius=25.4*u.mm/2).get_transmission(lwf)
        else:
            self.llowfsc_mode = False
            self.oversample = 2.5
            self.N = int(self.npix*self.oversample)
            lwf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, npix=self.npix, oversample=self.oversample)
            lyot = poppy.CircularAperture(radius=self.lyot_diam/2).get_transmission(lwf)
            if self.scc_diam is not None:
                scc_pinhole = poppy.CircularAperture(radius=self.scc_diam/2, 
                                                    shift_x=self.scc_pinhole_position[0], 
                                                    shift_y=self.scc_pinhole_position[1]).get_transmission(lwf)
                lyot += scc_pinhole
            self.LYOT = lyot

    def calc_wfs(self, save_wfs=True, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        wfs = []
        self.wf = utils.pad_or_crop(self.APERTURE, self.N).astype(complex)
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf *= utils.pad_or_crop(self.WFE, self.N)
        if save_wfs: wfs.append(copy.copy(self.wf))

        dm_surf = utils.pad_or_crop(self.DM.get_surface(), self.N)
        self.wf *= xp.exp(1j*4*np.pi/self.wavelength.to_value(u.m) * dm_surf)
        if save_wfs: wfs.append(copy.copy(self.wf))

        if self.return_pupil:
            return self.wf

        if self.use_fpm:
            self.wf = props.apply_vortex(utils.pad_or_crop(self.wf, self.Nfpm), npix=self.npix)
            self.wf = utils.pad_or_crop(self.wf, self.N)
        if save_wfs: wfs.append(copy.copy(self.wf))

        # self.wf *= utils.pad_or_crop(self.LYOT, self.N).astype(complex)
        # if self.reverse_parity: self.wf = xp.rot90(xp.rot90(self.wf))
        # if save_wfs: wfs.append(copy.copy(self.wf))

        if self.llowfsc_mode: 
            # propagate backwards to apply the 25.4mm aperture of the OAP
            if self.use_oap_ap:
                # imshows.imshow1(xp.abs(self.wf), 'WF at Lyot Pupil')
                self.wf = props.ang_spec(self.wf, self.wavelength, -450*u.mm, self.lyot_pupil_diam/(self.npix*u.pix))
                # imshows.imshow1(xp.abs(self.wf), 'WF at the OAP')

                oap_wf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, npix=self.npix, oversample=self.oversample)
                oap_ap = poppy.CircularAperture(radius=15*u.mm/2).get_transmission(oap_wf)
                self.wf *= oap_ap
                # imshows.imshow2(oap_ap, xp.abs(self.wf), 'OAP Aperture', 'WF after OAP')

                self.wf = props.ang_spec(self.wf, self.wavelength, 450*u.mm, self.lyot_pupil_diam/(self.npix*u.pix))
                if save_wfs: wfs.append(copy.copy(self.wf))
                # imshows.imshow1(xp.abs(self.wf), 'WF at the Lyot Pupil')

            # Apply the lyot stop
            self.wf *= utils.pad_or_crop(self.LYOT, self.N).astype(complex)
            if self.reverse_parity: self.wf = xp.rot90(xp.rot90(self.wf))
            if save_wfs: wfs.append(copy.copy(self.wf))

            # Use TF and MFT to propagate to defocused image
            fnum = self.llowfsc_fl.to_value(u.mm)/self.lyot_diam.to_value(u.mm)
            tf = props.get_fresnel_TF(self.llowfsc_defocus.to_value(u.m) * self.oversample**2, 
                                      self.N, self.wavelength.to_value(u.m), fnum)
            um_per_lamD = (self.wavelength * self.llowfsc_fl/self.lyot_diam).to(u.um)
            psf_pixelscale_lamD = (self.llowfsc_pixelscale.to(u.um/u.pix)/um_per_lamD).value
            self.llowfsc_pixelscale_lamD = psf_pixelscale_lamD
            self.wf = props. mft_forward(tf*self.wf, psf_pixelscale_lamD*self.oversample, self.nllowfsc)
            if save_wfs: wfs.append(copy.copy(self.wf))
            if save_wfs:
                return wfs
            else:
                return self.wf
            
        self.wf *= utils.pad_or_crop(self.LYOT, self.N).astype(complex)
        if self.reverse_parity: self.wf = xp.rot90(xp.rot90(self.wf))
        if save_wfs: wfs.append(copy.copy(self.wf))

        if self.use_fieldstop:
            self.wf = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(self.wf)))
            self.wf *= self.FIELDSTOP
            if save_wfs: wfs.append(copy.copy(self.wf))
            self.wf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(self.wf)))
            if save_wfs: wfs.append(copy.copy(self.wf))

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
    


