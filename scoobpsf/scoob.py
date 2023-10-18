import numpy as np
# import cupy as cp
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

from .math_module import xp,_scipy, ensure_np_array
from . import imshows
import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image


class SCOOBM():

    def __init__(self, 
                 wavelength=632.8e-9*u.m, 
                 npix=128, 
                 oversample=2048/128,
                 npsf=400, 
                 psf_pixelscale=4.63e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None,
                 Imax_ref=None,
                 det_rotation=0,
                 source_offset=(0,0),
                 use_synthetic_opds=False,
                 use_measured_opds=False,
                 use_noise=False, # FIXME: Noise simulations must be implemented, at the very least implement shot noise
                 use_pupil_grating=False,
                 use_aps=False,
                 inf_fun=None,
                 dm_ref=np.zeros((34,34)),
                 bad_acts=None,
                 OPD=None,
                 RETRIEVED=None,
                 ZWFS=None,
                 FPM=None,
                 LYOT=None,
                 pupil_diam=6.75*u.mm,
    ):
        
        '''
        Parameters
        ----------
        
        bad_acts: `list`
            List of 2d tuples that indicate where the bad actuators are. 
            For example [(23,25), (23,26)] would have bad actuators at Y=23 and X=25&26.
            Bad actuators are eventually masked in the Jacobian and when adding shapes.
            However, this aspect of the functionality is in the lina package.

        npsf: `int`
            Side length of the detector in pixels.

        npix: `int`
            Sampling rate at the pupil plane (TBR?)


        '''

        self.is_model = True
        
        self.pupil_diam = pupil_diam
        self.wavelength_c = wavelength
        
        if wavelength is None: 
            print('No wavelength provided on instantiation. '
                  f'Setting wavelength to {self.wavelength_c*1e9:0.1e} nm')
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.npix = int(npix)
        self.oversample = oversample
        self.N = int(self.npix*self.oversample)
        
        self.as_per_lamD = ((self.wavelength_c/self.pupil_diam)*u.radian).to(u.arcsec)
        self.source_offset = source_offset
        
        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = (1/(5)) * self.psf_pixelscale.to(u.m/u.pix).value/4.63e-6
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 4.63e-6*u.m/u.pix / self.psf_pixelscale_lamD/(1/5)
        self.norm = 'first' # This is the normalization that POPPY uses in propagating a wavefront
        
        self.det_rotation = det_rotation
        self.Imax_ref = Imax_ref
        
        self.use_synthetic_opds = use_synthetic_opds
        self.use_measured_opds = use_measured_opds
        self.use_opds = self.use_synthetic_opds or self.use_measured_opds
        self.init_opds()
        
        self.use_pupil_grating = use_pupil_grating
        self.use_aps = use_aps
        
        self.RETRIEVED = poppy.ScalarTransmission(name='Phase Retrieval Place-holder') if RETRIEVED is None else RETRIEVED
        self.FPM = poppy.ScalarTransmission(name='FPM Place-holder') if FPM is None else FPM
        self.LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if LYOT is None else LYOT
        
        self.ZWFS = ZWFS
        
        # Load influence function
        self.inf_fun = str(module_path/'inf.fits') if inf_fun is None else inf_fun
        self.bad_acts = bad_acts
        self.init_dm()
        self.dm_ref = ensure_np_array(dm_ref)
        self.set_dm(dm_ref)
        
    
    # useful for parallelization with ray actors
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

    def init_dm(self):
        self.Nact = 34
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        
        self.full_stroke = 1.5e-6*u.m
        
        self.dm_mask = np.ones((self.Nact,self.Nact), dtype=bool)

        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>10.5] = 0 # had to set the threshold to 10.5 instead of 10.2 to include edge actuators
        
        self.dm_bad_act_mask = copy.deepcopy(self.dm_mask)
        if self.bad_acts is not None:
            for act in self.bad_acts:
                self.dm_bad_act_mask[act] = False # bad actuators are False - a bit confusing but helps code

        self.dm_zernikes = ensure_np_array(poppy.zernike.arbitrary_basis(xp.array(self.dm_mask), nterms=15, outside=0))

        self.DM = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM', 
                                                   actuator_spacing=self.act_spacing, 
                                                   influence_func=self.inf_fun,
                                                  )
        
    def reset_dm(self):
        self.set_dm(self.dm_ref)
    
    def zero_dm(self):
        self.set_dm(np.zeros((34,34)))
        
    def set_dm(self, dm_command):
        self.DM.set_surface(ensure_np_array(dm_command))
        
    def add_dm(self, dm_command):
        self.DM.set_surface(self.get_dm() + ensure_np_array(dm_command))
        
    def get_dm(self):
        return ensure_np_array(self.DM.surface)
    
    def map_acts_to_dm(self, act_values):
        command = np.zeros((self.Nact, self.Nact))
        command.ravel()[self.dm_bad_act_mask.ravel()] = ensure_np_array(act_values)

        return command
    
    def map_dm_to_acts(self, command):
        acts = command[self.dm_mask]
        return acts
    
    def show_dm(self):
        wf = poppy.FresnelWavefront(beam_radius=self.dm_active_diam/2, npix=self.npix, oversample=1)
        dm_command = self.get_dm()
        dm_surface = self.DM.get_opd(wf).get() if poppy.accel_math._USE_CUPY else self.DM.get_opd(wf)
        surf_ext = wf.pixelscale.to_value(u.mm/u.pix)*self.npix/2
        
        imshows.imshow2(dm_command, dm_surface, 'DM Command', 'DM Surface')
    
    def init_inwave(self):
        self.pupil_diam = 6.8*u.mm
        
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        
        if self.source_offset[0]>0 or self.source_offset[1]>0:
            inwave.tilt(Xangle=self.source_offset[0]*self.as_per_lamD, Yangle=self.source_offset[1]*self.as_per_lamD)
        self.inwave = inwave
        
    def init_opds(self):
        if self.use_measured_opds:
            opd_dir = Path(os.path.dirname(str(module_path)))/'scoob-opds'
            
            try:
                self.m3_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'M3.fits'), opdunits='meters', planetype=inter)
                self.oap1_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP1.fits'), opdunits='meters', planetype=inter)
                self.oap2_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP2.fits'), opdunits='meters', planetype=inter)
                self.oap3_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP3.fits'), opdunits='meters', planetype=inter)
                self.flat1_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'FLAT1.fits'), opdunits='meters', planetype=inter)
                self.flat2_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'FLAT2.fits'), opdunits='meters', planetype=inter)
                print('Model using measured OPD data')
            except FileNotFoundError:
                print('Could not find OPD files. Must use synthetic OPD data instead.')
        elif self.use_synthetic_opds:
            s1, s2, s3, s4, s5, s6, s7 = (1,2,3,4,5,6,7)
            
            self.psd_index = 2.8
            self.opd_rms = 12*u.nm
            
            self.m3_diam = 25.4*u.mm
            self.flat1_diam = self.pupil_diam
            self.flat2_diam = 12.7*u.mm
            self.oap1_diam = 25.4*u.mm
            self.oap2_diam = 25.4*u.mm
            self.oap3_diam = 25.4*u.mm
            
            self.m3_opd = poppy.StatisticalPSDWFE(index=self.psd_index, wfe=self.opd_rms, radius=self.m3_diam/2, seed=s1)
            self.flat1_opd = poppy.StatisticalPSDWFE(index=self.psd_index, wfe=self.opd_rms, radius=self.flat1_diam/2, seed=s2)
            self.flat2_opd = poppy.StatisticalPSDWFE(index=self.psd_index, wfe=self.opd_rms, radius=self.flat2_diam/2, seed=s3)
            self.oap1_opd = poppy.StatisticalPSDWFE(index=self.psd_index, wfe=self.opd_rms, radius=self.oap1_diam/2, seed=s4)
            self.oap2_opd = poppy.StatisticalPSDWFE(index=self.psd_index, wfe=self.opd_rms, radius=self.oap2_diam/2, seed=s5)
            self.oap3_opd = poppy.StatisticalPSDWFE(index=self.psd_index, wfe=self.opd_rms, radius=self.oap3_diam/2, seed=s6)
            print('Model using synthetic OPD data')
        else:
            print('No OPD data implemented into model.')
    
    
    def init_pupil_grating(self):
        wf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=1)
        ap = poppy.CircularAperture(radius=self.pupil_diam/2).get_transmission(wf)
        
        grating_period = self.pupil_diam.to(u.m)/32
        grating_period_pix = grating_period.to_value(u.m)/wf.pixelscale.to_value(u.m/u.pix)
        grating_period_pix = round(grating_period_pix/2)*2

        gwf = poppy.FresnelWavefront(beam_radius=grating_period, npix=grating_period_pix, oversample=1)
        grating_obs = poppy.InverseTransmission(poppy.CircularAperture(radius=50*u.um)).get_transmission(gwf)

        nobs = self.npix//grating_period_pix

        grating = poppy.utils.pad_or_crop_to_shape(xp.tile(grating_obs, (nobs,nobs)), (self.npix, self.npix))*ap

        self.PUPIL_GRATING = poppy.ArrayOpticalElement(transmission=grating, pixelscale=wf.pixelscale)
        
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
        
        RETRIEVED = poppy.ScalarTransmission() if self.RETRIEVED is None else self.RETRIEVED
        FPM = poppy.ScalarTransmission() if self.FPM is None else self.FPM
        LYOT = poppy.ScalarTransmission() if self.LYOT is None else self.LYOT
        
        d_pupil_stop_flat = 42.522683120520355*u.mm
        d_flat_oap1 = 89.33864507546247*u.mm
        d_oap1_oap2 = 307.2973451416505*u.mm
        d_oap2_DM = 168.84723167004802*u.mm + 0*u.mm
        d_DM_oap3 = 460.43684751808945*u.mm
        d_oap3_FPM = 462.6680664924039*u.mm
        d_FPM_flat2 = 144.6375029385836*u.mm 
        d_flat2_lens = 200*u.mm - 144.6375029385836*u.mm
        d_lens_LYOT = 200*u.mm
        d_LYOT_scicam = 150*u.mm
        d_scicam_image = 150*u.mm

        fl_oap0 = self.oaefl(293.6,27)*u.mm
        fl_oap1 = self.oaefl(254,40)*u.mm
        fl_oap2 = self.oaefl(346,55)*u.mm
        fl_oap3 = self.oaefl(914.4,100)*u.mm
        
        fl_lens = 200*u.mm
        fl_scicam_lens = 150*u.mm
        
        # define optics 
        one_inch = poppy.CircularAperture(radius=25.4*u.mm/2, name='1in Aperture', gray_pixel=False)
        
        pupil_stop  = poppy.CircularAperture(radius=self.pupil_diam/2, name='Pupil Stop/FSM')
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
        self.N = int(self.npix*self.oversample)
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        
        fosys.add_optic(pupil_stop)
        if self.use_opds: fosys.add_optic(self.flat1_opd)
        if self.use_pupil_grating:
            self.init_pupil_grating()
            fosys.add_optic(self.PUPIL_GRATING)
        
        fosys.add_optic(flat1, distance=d_pupil_stop_flat)
        if self.use_opds: fosys.add_optic(self.flat2_opd)
        
        fosys.add_optic(oap1, distance=d_flat_oap1)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap1_opd)
        
        fosys.add_optic(oap2, distance=d_oap1_oap2)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap2_opd)
        
        fosys.add_optic(self.DM, distance=d_oap2_DM)
        if self.use_aps: fosys.add_optic(DM_aperture)
        
        fosys.add_optic(RETRIEVED)

        fosys.add_optic(oap3, distance=d_DM_oap3)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap3_opd)
        
        if self.ZWFS is not None:
            fosys.add_optic(self.ZWFS, distance=d_oap3_FPM)
            
            fl_zwfs_lens = 75*u.mm
            zwfs_lens = poppy.QuadraticLens(fl_zwfs_lens)
            fosys.add_optic(zwfs_lens, distance=fl_zwfs_lens)
            
            self.zwfs_pixelscale = 5*u.um/u.pix
            self.nzwfs = 400
            fosys.add_detector(pixelscale=self.zwfs_pixelscale.to(u.m/u.pix), fov_pixels=self.nzwfs, distance=fl_zwfs_lens)
        
            return fosys
        
        fosys.add_optic(FPM, distance=d_oap3_FPM)
        # fosys.add_optic(poppy.InverseTransmission(poppy.CircularAperture(radius=19*u.um/2)) )
        
        fosys.add_optic(flat2, distance=d_FPM_flat2)
        if self.use_aps: fosys.add_optic(one_inch)
        
        fosys.add_optic(lens, distance=d_flat2_lens)
        if self.use_aps: fosys.add_optic(one_inch)
        
        fosys.add_optic(lyot_plane, distance=d_lens_LYOT)
        fosys.add_optic(LYOT)
        
        fosys.add_optic(scicam_lens, distance=d_LYOT_scicam)
        if self.use_aps: fosys.add_optic(one_inch)
        
        fosys.add_detector(pixelscale=self.psf_pixelscale.to(u.m/u.pix), fov_pixels=self.npsf, distance=d_scicam_image)
        
        return fosys
    
    def calc_wfs(self, quiet=False):
        '''
        Propagate through the entire system and return the normalized wavefront
        for each surface.

        Parameters
        ----------

        quiet: `bool`
            Remove informational print statements.

        Returns
        -------

        wfs: `list`
            List of poppy.fresnel.FresnelWavefront objects corresponding to 
            each surface along the optical path.
        '''
        start = time.time()
        if not quiet: print(f'Propagating wavelength {self.wavelength.to(u.nm):.3f}.')
        fosys = self.init_fosys()
        self.init_inwave()
        _, wfs = fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_intermediates=True)
        if not quiet: print(f'PSF calculated in {(time.time()-start):.3f}s')
        
        # Rotate the detector image
        if abs(self.det_rotation)>0:
            rotated_wf = self.rotate_wf(wfs[-1])
            wfs[-1] = rotated_wf
        # Normalize the detector image
        wfs[-1].wavefront /= np.sqrt(self.imnorm)
        
        return wfs
    
    def calc_psf(self, plot=False,): 
        '''
        This propagates a beam from start to finish and returns the complex 
        wavefront.
        The calc_psf "notation" follows from poppy method naming.
        This is not actually the PSF (meaning modulus(wavefront)^2).
        
        The PSF will be normalized by whatever is in the self.imnorm class
        variable

        Parameters
        ----------

        plot: `bool`
            Plot result?

        Returns
        -------

        psf: `cupy.ndarray`
            Normalized (optional) wavefront array.

        '''

        fosys = self.init_fosys()
        self.init_inwave()
        _, wfs = fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_final=True, return_intermediates=False)

        wf = self.rotate_wf(wfs[-1]) if abs(self.det_rotation)>0 else wfs[-1]
        # Normalize by the unocculted PSF peak, however, this is a wavefront
        # and the peak is an intensity, so need to take the sqrt
        psf = wf.wavefront/np.sqrt(self.Imax_ref)
        if plot:
            imshows.imshow2(xp.abs(psf), xp.angle(psf), lognorm1=True, pxscl=self.psf_pixelscale_lamD)
        
        return psf
    
    def snap(self, plot=False):
        '''
        Calculate the and return the PSF intensity.

        Parameters
        ----------

        plot: `bool`
            Display a plot of the PSF?
        
        Returns
        -------

        im: `cupy.ndarray`
            PSF intensity on the focal plane.
        '''
        fosys = self.init_fosys()
        self.init_inwave()
        
        _, wfs = fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_intermediates=False, return_final=True)
        
        wf = self.rotate_wf(wfs[-1]) if abs(self.det_rotation)>0 else wfs[-1]
        im = wf.intensity
        
        if self.Imax_ref is not None:
            im *= 1/self.Imax_ref
            
        if plot:
            imshows.imshow1(im, lognorm=True, pxscl=self.psf_pixelscale_lamD)
            
        return im
    
    def rotate_wf(self, wave):
        wavefront = wave.wavefront
        wavefront_r = _scipy.ndimage.rotate(xp.real(wavefront), angle=-self.det_rotation, reshape=False, order=1)
        wavefront_i = _scipy.ndimage.rotate(xp.imag(wavefront), angle=-self.det_rotation, reshape=False, order=1)
        
        wave.wavefront = (wavefront_r + 1j*wavefront_i)
        return wave
    
