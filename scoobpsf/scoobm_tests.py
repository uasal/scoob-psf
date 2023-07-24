# What is this file? At first I thought it was unit tests but that doesn't appear to be true.

import numpy as np
import astropy.units as u
from astropy.constants import h, c
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

import cupy as cp
import cupyx.scipy.ndimage

class SCOOBM():

    def __init__(self, 
                 wavelength=None, 
                 npix=64, 
                 oversample=64,
                 npsf=64,
                 psf_pixelscale=4.63e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None,
                 interp_order=3,
                 det_rotation=0,
                 offset=(0,0),  
                 use_opds=False,
                 use_aps=False,
                 fpm_defocus=0,
                 dm_ref=np.zeros((34,34)),
                 dm_inf='inf.fits', # or 'proper_inf_func.fits'
                 im_norm=None,
                 OPD=None,
                 FPM=None,
                 LYOT=None):
        
        poppy.accel_math.update_math_settings()
        
        self.is_model = True
        
        self.wavelength_c = 632.8e-9*u.m
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.npix = npix
        self.oversample = oversample
        
        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = (1/2.75) * self.psf_pixelscale.to(u.m/u.pix).value/4.63e-6
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 4.63e-6*u.m/u.pix / self.psf_pixelscale_lamD/(1/2.75)
            
        self.interp_order = interp_order
        self.det_rotation = det_rotation
        
        self.dm_inf = dm_inf
        
        self.offset = offset
        self.use_opds = use_opds
        self.use_aps = use_aps
        self.fpm_defocus = fpm_defocus
        
        self.OPD = poppy.ScalarTransmission(name='OPD Place-holder') if OPD is None else OPD
        self.FPM = poppy.ScalarTransmission(name='FPM Place-holder') if FPM is None else FPM
        self.LYOT = poppy.ScalarTransmission(name='FPM Place-holder') if LYOT is None else LYOT
        
        self.texp = texp # between 0.1ms (0.0001s) and 0.01s
        
        self.im_norm = im_norm
        
        self.init_dm()
        self.dm_ref = dm_ref
        if self.use_opds:
            self.init_opds()
        
    def getattr(self, attr):
        return getattr(self, attr)

    def init_dm(self):
        self.Nact = 34
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        
        self.full_stroke = 1.5e-6*u.m
        
        self.dm_mask = np.ones((self.Nact,self.Nact))
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>10.5] = 0
        
        self.dm_zernikes = poppy.zernike.arbitrary_basis(cp.array(self.dm_mask), nterms=15, outside=0).get()
        
        bad_acts = [(21,25)]
        self.bad_acts = []
        for act in bad_acts:
            self.bad_acts.append(act[1]*self.Nact + act[0])
        
        self.DM = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM', 
                                                   actuator_spacing=self.act_spacing, 
                                                   influence_func=str(esc_coro_suite.data_dir/self.dm_inf),
                                                  )
        
    def reset_dm(self):
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm(self, dm_command):
        self.DM.set_surface(dm_command)
        
    def add_dm(self, dm_command):
        self.DM.set_surface(self.get_dm() + dm_command)
        
    def get_dm(self):
        return self.DM.surface.get()
    
    def oaefl(self, roc, oad, k=-1):
        """
        roc: float
            parent parabola radius of curvature
        angle: float
            off-axis angle in radians
        oad: float
            off-axis distance
        """
        # compute parabolic sag
        sag = (1/roc)*oad**2 /(1 + np.sqrt(1-(k+1)*(1/roc)**2 * oad**2))
        return roc/2 + sag
    
    def init_fosys(self):
        d_pinhole_oap0 = 150.48644122948647*u.mm
        d_oap0_fsm = 145.538687829754*u.mm
        d_fsm_flat = 42.522683120520355*u.mm
        d_flat_oap1 = 89.33864507546247*u.mm
        d_oap1_oap2 = 307.2973451416505*u.mm
        d_oap2_DM = 168.84723167004802*u.mm
#         d_DM_oap3 = 460.43684751808945*u.mm
        d_DM_oap3 = 462.6680664924039*u.mm
#         d_oap3_FPM = 462.6680664924039*u.mm - 12*u.mm + 1.304*u.mm + self.fpm_defocus # ZEMAX required defocus term of 7mm
#         d_FPM_flat2 = 144.6375029385836*u.mm - 1.304*u.mm - self.fpm_defocus
        d_oap3_FPM = 462.6680664924039*u.mm - 12*u.mm + self.fpm_defocus # ZEMAX required defocus term of 7mm
        d_FPM_flat2 = 144.6375029385836*u.mm - self.fpm_defocus
        d_flat2_lens = 200*u.mm - 144.6375*u.mm
        d_lens_LYOT = 200*u.mm
#         d_LYOT_scicam = 75*u.mm
#         d_scicam_image = 75*u.mm
        d_LYOT_scicam = 150*u.mm
        d_scicam_image = 150*u.mm
        
        fl_oap0 = self.oaefl(293.6,27)*u.mm
        fl_oap1 = self.oaefl(254,40)*u.mm
        fl_oap2 = self.oaefl(346,55)*u.mm
        fl_oap3 = self.oaefl(914.4,100)*u.mm
        fl_lens = 200*u.mm
#         fl_scicam_lens = 75*u.mm
        fl_scicam_lens = 150*u.mm
        
        # define optics 
        one_inch = poppy.CircularAperture(radius=25.4*u.mm/2, name='1in Aperture', gray_pixel=False)
        half_inch = poppy.CircularAperture(radius=25.4/2*u.mm/2, name='1in Aperture', gray_pixel=False)
        
        oap0 = poppy.QuadraticLens(fl_oap0, name='OAP0')
        pupil_stop  = poppy.CircularAperture(radius=6.8*u.mm/2, name='Pupil Stop/FSM')
        flat1 = poppy.CircularAperture(radius=12.7*u.mm/2, name='Flat 1')
        oap1 = poppy.QuadraticLens(fl_oap1, name='OAP1')
        oap2 = poppy.QuadraticLens(fl_oap2, name='OAP2')
        DM_aperture = poppy.SquareAperture(size=self.dm_full_diam, name='DM aperture')
        oap3 = poppy.QuadraticLens(fl_oap3, name='OAP3')
        flat2 = poppy.CircularAperture(radius=25.4*u.mm/2, name='Flat 2')
        lens = poppy.QuadraticLens(fl_lens, name='Lens')
        lyot_plane = poppy.ScalarTransmission(name='Lyot plane')
        scicam_lens = poppy.QuadraticLens(fl_scicam_lens, name='Science Lens')
        
        # define FresnelOpticalSystem and add optics
#         fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
#         fiber_input = poppy.CompoundAnalyticOptic(opticslist=[poppy.GaussianAperture(fwhm=2*0.85*self.inwave.w_0),
#                                                               poppy.CircularAperture(radius=self.inwave.w_0*4),
#                                                              ])
#         fosys.add_optic(fiber_input)
        
#         fosys.add_optic(oap0, distance=d_pinhole_oap0-0*u.mm)
#         fosys.add_optic(half_inch)
#         if self.use_opds: fosys.add_optic(self.m3_opd)
        
#         fosys.add_optic(pupil_stop, distance=d_oap0_fsm)
#         if self.use_opds: fosys.add_optic(self.flat1_opd)
#         fosys.add_optic(self.OPD)
        
        self.pupil_diam = 6.8*u.mm
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        fosys.add_optic(pupil_stop)
        if self.use_opds: fosys.add_optic(self.flat1_opd)
        fosys.add_optic(self.OPD)

        fosys.add_optic(flat1, distance=d_fsm_flat)
        if self.use_opds: fosys.add_optic(self.flat2_opd)
        
        fosys.add_optic(oap1, distance=d_flat_oap1)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap1_opd)
        
        fosys.add_optic(oap2, distance=d_oap1_oap2)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap2_opd)
        
#         fosys.add_optic(poppy.ScalarTransmission(name='DM placeholder'), distance=d_oap2_DM)
        fosys.add_optic(self.DM, distance=d_oap2_DM)
        if self.use_aps: fosys.add_optic(DM_aperture)
#         if self.use_opds: fosys.add_optic(self.dm_opd)

        fosys.add_optic(oap3, distance=d_DM_oap3)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap3_opd)
        
        fosys.add_optic(self.FPM, distance=d_oap3_FPM)
        
        fosys.add_optic(flat2, distance=d_FPM_flat2)
        if self.use_aps: fosys.add_optic(one_inch)
        
        fosys.add_optic(lens, distance=d_flat2_lens)
        if self.use_aps: fosys.add_optic(one_inch)
        
        fosys.add_optic(lyot_plane, distance=d_lens_LYOT)
        fosys.add_optic(self.LYOT)
        
        fosys.add_optic(scicam_lens, distance=d_LYOT_scicam)
        if self.use_aps: fosys.add_optic(one_inch)
        
        fosys.add_optic(poppy.ScalarTransmission(), distance=d_scicam_image)
        
        self.fosys = fosys
        
    def init_opds(self):
        opd_dir = esc_coro_suite.data_dir/'scoob-opds'
        
        self.m3_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'M3.fits'), opdunits='meters', planetype=inter)
        self.oap1_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP1.fits'), opdunits='meters', planetype=inter)
        self.oap2_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP2.fits'), opdunits='meters', planetype=inter)
        self.oap3_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP3.fits'), opdunits='meters', planetype=inter)
        self.flat1_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'FLAT1.fits'), opdunits='meters', planetype=inter)
        self.flat2_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'FLAT2.fits'), opdunits='meters', planetype=inter)
    
    def init_inwave(self):
        self.pupil_diam = 0.020*u.mm
        
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        
        inwave.w_0 = 5e-6*u.m/2
#         inwave.w_0 = 5e-6*u.m
        
        self.inwave = inwave
    
    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_inwave()
        self.init_fosys()
        psf, wfs = self.fosys.calc_psf(inwave=self.inwave, return_intermediates=True)
#         psf, wfs = self.fosys.calc_psf(return_intermediates=True)

        if not self.use_opds and not self.use_aps:
            self.fosys_names = [
                                'Fiber Input', 
                                'OAP0/M3', 'OAP0 Aperture',
                                'Pupil Stop', 'Injected OPD', 'Flat 1', 'OAP1', 'OAP2',
                                'DM', 'OAP3',
                                'FPM', 'Flat 2', 'Lens',  
                                'Lyot Plane', 'Lyot Stop',
                                'SciCam Lens', 'Image Plane']
        elif self.use_opds and not self.use_aps:
            self.fosys_names = ['Fiber Input', 
                                'OAP0/M3', 'OAP0 Aperture', 'M3 OPD',
                                'Pupil Stop', 'Flat 1 OPD', 'Injected OPD', 'Flat 1', 'Flat 2 OPD', 
                                'OAP1', 'OAP1 OPD', 'OAP2', 'OAP2 OPD',
                                'DM', 'OAP3', 'OAP3 OPD',
                                'FPM', 'Flat 2', 'Lens',  
                                'Lyot Plane', 'Lyot Stop',
                                'SciCam Lens', 'Image Plane']
            
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        return wfs
    
    def calc_psf(self, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_inwave()
        self.init_fosys()
        psf, wfs = self.fosys.calc_psf(inwave=self.inwave, return_final=True, return_intermediates=False)

        if self.im_norm is not None:
            wfs[-1].wavefront *= np.sqrt(self.im_norm)/abs(wfs[-1].wavefront).max()
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        wavefront = wfs[-1].wavefront
        wavefront_r = cupyx.scipy.ndimage.rotate(cp.real(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        wavefront_i = cupyx.scipy.ndimage.rotate(cp.imag(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        
        wfs[-1].wavefront = wavefront_r + 1j*wavefront_i
        
        resamped_wf = self.interp_wf(wfs[-1])
        
        return resamped_wf.get()
    
    def snap(self): # method for getting the PSF in photons
        self.init_inwave()
        self.init_fosys()
        psf, wfs = self.fosys.calc_psf(inwave=self.inwave, return_intermediates=False, return_final=True)
        
        wavefront = wfs[-1].wavefront
        wavefront_r = cupyx.scipy.ndimage.rotate(cp.real(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        wavefront_i = cupyx.scipy.ndimage.rotate(cp.imag(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        
        wfs[-1].wavefront = wavefront_r + 1j*wavefront_i
        
        resamped_wf = self.interp_wf(wfs[-1])
        
        image = (cp.abs(resamped_wf)**2).get()
        
        if self.im_norm is not None:
            image = image/image.max() * self.im_norm
        
        return image
    
    def interp_wf(self, wave): # this will interpolate the FresnelWavefront data to match the desired pixelscale
        n = wave.wavefront.shape[0]
        xs = (cp.linspace(0, n-1, n))*wave.pixelscale.to(u.m/u.pix).value
        
        extent = self.npsf*self.psf_pixelscale.to(u.m/u.pix).value
        for i in range(n):
            if xs[i+1]>extent:
                newn = i
                break
        newn += 2
        cropped_wf = misc.pad_or_crop(wave.wavefront, newn)

        wf_xmax = wave.pixelscale.to(u.m/u.pix).value * newn/2
        x,y = cp.ogrid[-wf_xmax:wf_xmax:cropped_wf.shape[0]*1j,
                       -wf_xmax:wf_xmax:cropped_wf.shape[1]*1j]

        det_xmax = extent/2
        newx,newy = cp.mgrid[-det_xmax:det_xmax:self.npsf*1j,
                               -det_xmax:det_xmax:self.npsf*1j]
        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0] - x0
        dy = y[0,1] - y0

        ivals = (newx - x0)/dx
        jvals = (newy - y0)/dy

        coords = cp.array([ivals, jvals])

        resamped_wf = cupyx.scipy.ndimage.map_coordinates(cropped_wf, coords, order=3)
        m = (wave.pixelscale.to(u.m/u.pix)/self.psf_pixelscale.to(u.m/u.pix)).value
        resamped_wf /= m
        
        return resamped_wf
    

