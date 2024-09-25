from .math_module import xp, _scipy, ensure_np_array
import scoobpsf.utils as utils
import scoobpsf.imshows as imshows
import scoobpsf.props as props
import scoobpsf.dm as dm

import numpy as np
import astropy.units as u
import copy
import poppy
from scipy.signal import windows

class single():

    def __init__(self,
                 wavelength=633e-9*u.m, 
                 entrance_flux=None, 
                 scc_sep=(1.55/np.sqrt(2) * 3.7*u.mm, 1.55/np.sqrt(2) * 3.7*u.mm),
                 scc_diam=100*u.um, 
                 ):
        self.wavelength_c = 633e-9*u.m
        self.dm_beam_diam = 9.2*u.mm # as measured in the Fresnel model
        self.lyot_pupil_diam = 9.2*u.mm
        self.lyot_diam = 8.7*u.mm
        self.lyot_ratio = (self.lyot_diam/self.lyot_pupil_diam).decompose().value
        self.rls_diam = 25.4*u.mm
        self.imaging_fl = 140*u.mm
        self.llowfsc_fl = 200*u.mm
        self.llowfsc_defocus = 1.75*u.mm
        self.psf_pixelscale = 3.76*u.um / u.pix
        self.psf_pixelscale_lamDc = 0.307
        self.llowfsc_pixelscale = 3.76*u.um / u.pix
        self.llowfsc_pixelscale_lamDc = (self.psf_pixelscale / (self.llowfsc_fl * self.wavelength_c / self.lyot_pupil_diam)).decompose().value

        self.wavelength = wavelength
        self.use_vortex = False
        self.plot_vortex = False

        self.scc_diam = scc_diam
        self.scc_sep = scc_sep
        
        self.npix = 1000
        self.def_oversample = 2.048 # default oversample
        self.rls_oversample = 3 # reflective lyot stop oversample
        self.Ndef = int(self.npix*self.def_oversample)
        self.Nrls = int(self.npix*self.rls_oversample)
        self.npsf = 150
        self.nlocam = 64

        ### INITIALIZE APERTURES ###
        pwf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=1)
        scc_wf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=self.def_oversample) 
        rls_wf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=self.rls_oversample)
        self.APERTURE = poppy.CircularAperture(radius=self.dm_beam_diam/2).get_transmission(pwf)
        self.LYOTSTOP = poppy.CircularAperture(radius=self.lyot_diam/2).get_transmission(pwf)
        self.MSCC_PINHOLE = poppy.CircularAperture(radius=self.scc_diam/2, shift_x=self.scc_sep[0], shift_y=self.scc_sep[1]).get_transmission(scc_wf)
        self.LYOT_MSCC = utils.pad_or_crop(self.LYOTSTOP, self.Ndef) + self.MSCC_PINHOLE
        rls_ap = poppy.CircularAperture(radius=self.rls_diam/2).get_transmission(rls_wf)
        self.RLS = rls_ap - utils.pad_or_crop( self.LYOTSTOP, self.Nrls) - utils.pad_or_crop( self.MSCC_PINHOLE, self.Nrls)
        rls_ap = 0
        self.use_locam = False

        self.LYOT = self.LYOTSTOP
        self.oversample = self.def_oversample
        self.N = self.Ndef # default to not using RLS

        self.APMASK = self.APERTURE>0
        self.WFE = xp.ones((self.npix,self.npix), dtype=complex)

        self.Imax_ref = 1
        self.entrance_flux = entrance_flux
        if self.entrance_flux is not None:
            pixel_area = (self.pupil_diam/self.npix)**2
            flux_per_pixel = self.entrance_flux * pixel_area
            self.APERTURE *= xp.sqrt(flux_per_pixel.to_value(u.photon/u.second))

        ### INITIALIZE DM PARAMETERS ###
        self.Nact = 34
        act_spacing = 300e-6*u.m
        dm_pxscl = self.dm_beam_diam.to_value(u.m)/self.npix
        inf_sampling = act_spacing.to_value(u.m)/dm_pxscl
        inf_fun = dm.make_gaussian_inf_fun(act_spacing=act_spacing, sampling=inf_sampling, coupling=0.15, Nact=self.Nact+2)
        self.DM = dm.DeformableMirror(inf_fun=inf_fun, inf_sampling=inf_sampling, name='DM (pupil)')
        self.dm_mask = self.DM.dm_mask
        self.Nacts = self.DM.Nacts
        self.dm_ref = xp.zeros((self.Nact, self.Nact))

        ### INITIALIZE VORTEX PARAMETERS ###
        self.oversample_vortex = 4.096
        self.N_vortex_lres = int(self.npix*self.oversample_vortex)
        self.vortex_win_diam = 30 # diameter of the window to apply with the vortex model
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(self.vortex_win_diam/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(self.vortex_win_diam/self.hres_sampling))
        self.hres_win_size = int(self.vortex_win_diam/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2) * self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        self.hres_dot_mask = r>=0.15

    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc * (self.wavelength_c/wl).decompose().value
        self.llowfsc_pixelscale_lamD = self.llowfsc_pixelscale_lamDc * (self.wavelength_c/wl).decompose().value

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

    def use_llowfsc(self, use=True):
        if use:
            self.use_locam = True
            self.N = self.Nrls
            self.oversample = self.rls_oversample
            self.LYOT = self.RLS
        else:
            self.use_locam = False
            self.N = self.Ndef
            self.oversample = self.def_oversample
            self.LYOT = self.LYOTSTOP
        

    def apply_vortex(self, pupwf, plot=False):
        lres_wf = utils.pad_or_crop(pupwf, self.N_vortex_lres) # pad to the larger array for the low res propagation
        fp_wf_lres = props.fft(lres_wf)
        fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
        pupil_wf_lres = props.ifft(fp_wf_lres)
        pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N,)
        if plot: 
            imshows.imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 
                            'FFT Pupil Amplitude', 'FFT Pupil Phase', 
                            npix=int(1.5*self.npix), cmap2='twilight', 
                            )

        fp_wf_hres = props.mft_forward(pupwf, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
        fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res (windowed) FPM
        pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+')
        if plot: 
            imshows.imshow2(xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 
                            'MFT Pupil Amplitude', 'MFT Pupil Phase',
                            npix=int(1.5*self.npix), cmap2='twilight', 
                            )

        post_vortex_pup_wf = (pupil_wf_lres + pupil_wf_hres)
        if plot: 
            imshows.imshow2(xp.abs(post_vortex_pup_wf), xp.angle(post_vortex_pup_wf), 
                            'Total Pupil Amplitude', 'Total Pupil Phase',
                            npix=int(1.5*self.npix), cmap2='twilight', 
                            )

        return post_vortex_pup_wf

    def calc_wfs(self, save_wfs=True, plot=False): # method for getting the PSF in photons
        wfs = []
        wf = self.APERTURE.astype(complex)
        if save_wfs: wfs.append(copy.copy(wf))

        wf *= self.WFE
        if save_wfs: wfs.append(copy.copy(wf))
        if plot: imshows.imshow2(xp.abs(wf), xp.angle(wf), cmap2='twilight', npix=int(1.5*self.npix))

        dm_surf = utils.pad_or_crop(self.DM.get_surface(), self.npix)
        wf *= xp.exp(1j*4*np.pi*self.wavelength.to_value(u.m) * dm_surf)
        if save_wfs: wfs.append(copy.copy(wf))
        if plot: imshows.imshow2(xp.abs(wf), xp.angle(wf), cmap2='twilight', npix=int(1.5*self.npix))

        if self.use_vortex:
            wf = self.apply_vortex(wf, plot=plot)
        if save_wfs: wfs.append(copy.copy(wf))
        print(wf.shape)

        wf *= utils.pad_or_crop(self.LYOT, wf.shape[0]).astype(complex)
        if save_wfs: wfs.append(copy.copy(wf))
        if plot: imshows.imshow2(xp.abs(wf), xp.angle(wf), cmap2='twilight')

        if self.use_locam:
            # Use TF and MFT to propagate to defocused image
            fnum = self.llowfsc_fl.to_value(u.mm)/self.lyot_diam.to_value(u.mm)
            tf = props.get_fresnel_TF(self.llowfsc_defocus.to_value(u.m) * self.rls_oversample**2, 
                                      self.Nrls, 
                                      self.wavelength.to_value(u.m), 
                                      fnum)
            wf = props.mft_forward(tf*wf, self.npix*self.lyot_ratio, self.nlocam, self.llowfsc_pixelscale_lamD)
            if plot: imshows.imshow2(xp.abs(wf)**2, xp.angle(wf), cmap2='twilight',)
            if save_wfs: 
                wfs.append(copy.copy(wf))
                return wfs
            else:
                return wf

        wf = props.mft_forward(wf, self.npix*self.lyot_ratio, self.npsf, self.psf_pixelscale_lamD)
        wf /= xp.sqrt(self.Imax_ref) # normalize by a reference maximum value
        if save_wfs: wfs.append(copy.copy(wf))
        if plot: imshows.imshow2(xp.abs(wf)**2, xp.angle(wf), cmap2='twilight',)

        if save_wfs:
            return wfs
        else:
            return wf
    
    def calc_wf(self):
        fpwf = self.calc_wfs(save_wfs=False)
        return fpwf
    
    def snap(self):
        image = xp.abs(self.calc_wfs(save_wfs=False))**2
        return image

class multi():
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
        
        self.psf_pixelscale = ray.get(actors[0].getattr.remote('psf_pixelscale'))
        self.psf_pixelscale_lamD = ray.get(actors[0].getattr.remote('psf_pixelscale_lamDc'))
        self.npsf = ray.get(actors[0].getattr.remote('npsf'))

        self.dm_mask = ray.get(actors[0].getattr.remote('dm_mask'))
        self.Nact = self.dm_mask.shape[0]
        self.dm_ref = ray.get(actors[0].getattr.remote('dm_ref'))

        self.Imax_ref = 1

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

    def snap(self):
        pending_ims = []
        for i in range(self.Nactors):
            future_ims = self.actors[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        im = xp.mean(ims, axis=0)

        return im/self.Imax_ref
    


