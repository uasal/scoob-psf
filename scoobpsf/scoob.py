import numpy as np
# import cupy as cp
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

from .math_module import xp,_scipy, ensure_np_array
from .import dm, optics, opds, utils, imshows

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
                 npix=256, 
                 oversample=4,
                 use_llowfsc=False,
                 npsf=256, 
                 psf_pixelscale=3.76*u.um/u.pix,
                 psf_pixelscale_lamD=None,
                 Imax_ref=1,
                 det_rotation=0,
                 source_offset=(0,0),
                 use_synthetic_opds=False,
                 use_measured_opds=False,
                 use_noise=False, # FIXME: Noise simulations must be implemented, at the very least implement shot noise
                 use_pupil_grating=False,
                 use_aps=False,
                 inf_fun=None,
                 dm_ref=xp.zeros((34,34)),
                 bad_acts=None,
                 WFE=None,
                 ZWFS=None,
                 FPM=None,
                 LYOT=None,
                 FIELDSTOP=None,
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
        
        # focal lengths of powered optics 
        self.fl_oap1 = 254/2*u.mm
        self.fl_oap2 = 346/2*u.mm
        self.fl_oap3 = 914.4/2*u.mm
        self.fl_oap4 = 914/2*u.mm
        self.fl_oap5 = 346/2*u.mm
        self.fl_oap6 = 254/2*u.mm
        self.fl_oap7 = 609.6/2*u.mm

        self.d_stop_oap1 = self.fl_oap1
        self.d_oap1_ifp1 = self.fl_oap1
        self.d_ifp1_oap2 = self.fl_oap2
        self.d_oap2_DM = self.fl_oap2
        self.d_DM_oap3 = self.fl_oap3 # I need to actually place the LP and QWP before OAP3
        self.d_oap3_FPM = self.fl_oap3
        self.d_FPM_flat1 = 257*u.mm
        self.d_flat1_oap4 = self.fl_oap4 - self.d_FPM_flat1
        self.d_oap4_flat2 = 250*u.mm
        self.d_flat2_LYOT = self.fl_oap4 - self.d_oap4_flat2
        self.d_LYOT_oap5 = self.fl_oap5
        self.d_oap5_ifp2 = self.fl_oap5
        self.d_ifp2_oap6 = self.fl_oap6
        self.d_oap6_oap7 = self.fl_oap6+self.fl_oap7
        # self.d_oap6_oap7 = 481*u.mm
        self.d_oap7_CAM = self.fl_oap7

        self.flat1_diam = 25.4*u.mm
        self.flat2_diam = 25.4*u.mm
        self.oap1_diam = 25.4*u.mm
        self.oap2_diam = 25.4*u.mm
        self.oap3_diam = 25.4*u.mm

        self.npix = int(npix)
        self.oversample = oversample
        self.N = int(self.npix*self.oversample)
        
        self.as_per_lamD = ((self.wavelength_c/self.pupil_diam)*u.radian).to(u.arcsec)
        self.source_offset = source_offset
        
        self.npsf = npsf
        self.lyot_pupil_diam = 9.19*u.mm
        self.lyot_stop_diam = 8.6*u.mm

        self.um_per_lamD = (self.fl_oap7*self.wavelength_c/(self.lyot_stop_diam)).to(u.um)
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD =  self.psf_pixelscale.to_value(u.um/u.pix) / self.um_per_lamD.value
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = self.psf_pixelscale_lamD * self.um_per_lamD/u.pix
        self.norm = 'first' # This is the normalization that POPPY uses in propagating a wavefront
        
        self.det_rotation = det_rotation
        self.Imax_ref = Imax_ref
        
        self.use_pupil_grating = use_pupil_grating
        self.use_aps = use_aps
        wf = poppy.Wavefront(diam=self.pupil_diam, npix=self.npix, oversample=1)
        self.pupil_mask = poppy.CircularAperture(radius=self.pupil_diam/2, name='Pupil').get_transmission(wf).astype(bool)

        self.WFE = poppy.ScalarTransmission(name='Input WFE Place-holder') if WFE is None else WFE
        self.FPM = poppy.ScalarTransmission(name='Focal Plane Mask Place-holder') if FPM is None else FPM
        self.LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if LYOT is None else LYOT
        self.FIELDSTOP = poppy.ScalarTransmission(name='Field stop Place-holder') if FIELDSTOP is None else FIELDSTOP
        
        self.ZWFS = ZWFS
        
        self.use_llowfsc = use_llowfsc
        self.llowfsc_pixelscale = 3.76*u.um/u.pix # Sony IMX571
        self.nllowfsc = 64
        self.llowfsc_defocus = 0.9*u.mm
        self.fl_llowfsc_lens = 100*u.mm

        self.distances = {
            ### FIXME: we should implement the distances as a dictionary that gets loaded in by a toml file
        }

        self.diams = {
            ### FIXME
        }

        self.psd_index = 2.75
        self.opd_rms = 12*u.nm
        self.use_synthetic_opds = use_synthetic_opds
        self.use_measured_opds = use_measured_opds
        self.use_opds = self.use_synthetic_opds or self.use_measured_opds
        self.init_opds()

        # Load influence function
        self.inf_fun = str(module_path/'inf.fits') if inf_fun is None else inf_fun
        self.bad_acts = bad_acts
        self.init_dm()

        self.dm_ref = dm_ref
        self.set_dm(dm_ref)
        
        self.return_pupil = False
        self.return_lyot = False
            
    # useful for parallelization with ray actors
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

    def init_dm(self):
        self.Nact = 34
        self.dm_sampling = self.npix/self.Nact # pixels per actuator
        self.DM = dm.DeformableMirror(name='DM (Pupil)', inf_sampling=self.dm_sampling)

        self.Nacts = self.DM.Nacts
        self.act_spacing = self.DM.act_spacing
        self.dm_active_diam = self.DM.active_diam
        self.dm_full_diam = self.DM.pupil_diam
        
        self.full_stroke = self.DM.full_stroke

        self.dm_mask = self.DM.dm_mask
        
    def reset_dm(self):
        self.set_dm(self.dm_ref)
    
    def zero_dm(self):
        # self.DM.set_surface(np.zeros((34,34)))
        self.DM.command = xp.zeros((self.Nact,self.Nact))
        
    def set_dm(self, dm_command):
        # self.DM.set_surface(ensure_np_array(dm_command))
        self.DM.command = dm_command
        
    def add_dm(self, dm_command):
        # self.DM.set_surface(ensure_np_array(self.get_dm()) + ensure_np_array(dm_command))
        self.DM.command = self.get_dm() + dm_command
        
    def get_dm(self):
        # return self.DM.surface
        return self.DM.command
    
    def map_acts_to_dm(self, act_values):
        command = xp.zeros((self.Nact, self.Nact))
        command.ravel()[self.dm_bad_act_mask.ravel()] = act_values

        return command
    
    def map_dm_to_acts(self, command):
        acts = command[self.dm_mask]
        return acts
        
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
            
            # self.m3_opd = poppy.StatisticalPSDWFE(index=self.psd_index, wfe=self.opd_rms, radius=self.m3_diam/2, seed=s1)
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
        oad: float
            off-axis distance
        """
        # compute parabolic sag
        sag = (1/roc)*oad**2 /(1 + np.sqrt(1-(k+1)*(1/roc)**2 * oad**2))
        return roc/2 + sag
    
    def init_fosys(self):
        
        WFE = poppy.ScalarTransmission() if self.WFE is None else self.WFE
        FPM = poppy.ScalarTransmission() if self.FPM is None else self.FPM
        LYOT = poppy.ScalarTransmission() if self.LYOT is None else self.LYOT
        FIELDSTOP = poppy.ScalarTransmission() if self.FIELDSTOP is None else self.FIELDSTOP

        # define optics 
        one_inch = poppy.CircularAperture(radius=25.4*u.mm/2, name='1in Aperture',)
        
        pupil_stop = poppy.CircularAperture(radius=self.pupil_diam/2, name='Pupil Stop/FSM')
        oap1 = poppy.QuadraticLens(self.fl_oap1, name='OAP1')
        oap2 = poppy.QuadraticLens(self.fl_oap2, name='OAP2')
        oap3 = poppy.QuadraticLens(self.fl_oap3, name='OAP3')
        oap4 = poppy.QuadraticLens(self.fl_oap4, name='OAP4')
        oap5 = poppy.QuadraticLens(self.fl_oap5, name='OAP5')
        oap6 = poppy.QuadraticLens(self.fl_oap6, name='OAP6')
        oap7 = poppy.QuadraticLens(self.fl_oap7, name='OAP7')
        
        ifp1 = poppy.ScalarTransmission(name='IFP1')
        fpm_plane = poppy.ScalarTransmission(name='FPM')
        fielstop_plane = poppy.ScalarTransmission(name='Fieldstop FP')

        flat1 = poppy.CircularAperture(radius=25.4*u.mm/2, name='Fold Flat 1',)
        flat2 = poppy.CircularAperture(radius=25.4*u.mm/2, name='Fold Flat 2',)

        lyot_plane = poppy.ScalarTransmission(name='Lyot Pupil')
        
        # define FresnelOpticalSystem and add optics
        self.N = int(self.npix*self.oversample)
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        fosys.add_optic(pupil_stop)
        fosys.add_optic(WFE)
        if self.use_opds: fosys.add_optic(self.flat1_opd)
        if self.use_pupil_grating:
            self.init_pupil_grating()
            fosys.add_optic(self.PUPIL_GRATING)
        fosys.add_optic(oap1, distance=self.d_stop_oap1)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap1_opd)
        fosys.add_optic(ifp1, distance=self.d_oap1_ifp1)
        fosys.add_optic(oap2, distance=self.d_ifp1_oap2)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap2_opd)
        fosys.add_optic(self.DM, distance=self.d_oap2_DM)
        if self.use_aps: fosys.add_optic(DM_aperture)
        if self.return_pupil:
            return fosys
        fosys.add_optic(oap3, distance=self.d_DM_oap3)
        if self.use_aps: fosys.add_optic(one_inch)
        if self.use_opds: fosys.add_optic(self.oap3_opd)
        fosys.add_optic(fpm_plane, distance=self.d_oap3_FPM)
        fosys.add_optic(FPM,)
        fosys.add_optic(flat1, distance=self.d_FPM_flat1)
        if self.use_aps: fosys.add_optic(one_inch)
        fosys.add_optic(oap4, distance=self.d_flat1_oap4)
        fosys.add_optic(flat2, distance=self.d_oap4_flat2)
        fosys.add_optic(lyot_plane, distance=self.d_flat2_LYOT)
        if self.return_lyot:
            return fosys
        fosys.add_optic(LYOT)
        if self.use_llowfsc:
            llowfsc_lens = poppy.QuadraticLens(self.fl_llowfsc_lens, name='Lens')
            fosys.add_optic(llowfsc_lens, distance=self.fl_llowfsc_lens)
            fosys.add_optic(poppy.Detector(pixelscale=self.llowfsc_pixelscale, fov_pixels=self.nllowfsc, interp_order=3, name='LLOWFSC Detector'),
                            distance=self.fl_llowfsc_lens + self.llowfsc_defocus)
            return fosys
        fosys.add_optic(oap5, distance=self.d_LYOT_oap5)
        fosys.add_optic(fielstop_plane, distance=self.d_oap5_ifp2)
        fosys.add_optic(FIELDSTOP,)
        fosys.add_optic(oap6, distance=self.d_ifp2_oap6)
        fosys.add_optic(oap7, distance=self.d_oap6_oap7)
        fosys.add_optic(poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=3, name='Camera FP'),
                        distance=self.d_oap7_CAM)

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
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        if self.source_offset[0]>0 or self.source_offset[1]>0:
            inwave.tilt(Xangle=self.source_offset[0]*self.as_per_lamD, Yangle=self.source_offset[1]*self.as_per_lamD)
        
        _, wfs = fosys.calc_psf(inwave=inwave, normalize='none', return_intermediates=True)
        if not quiet: print(f'PSF calculated in {(time.time()-start):.3f}s')
        
        # Normalize the detector image
        wfs[-1].wavefront /= np.sqrt(self.Imax_ref)
        
        return wfs
    
    def calc_wf(self, pixelscale=False): 
        '''
        This propagates a beam from start to finish and returns the complex 
        wavefront.
        The calc_psf "notation" follows from poppy method naming.
        This is not actually the PSF (meaning modulus(wavefront)^2).
        
        The PSF will be normalized by whatever is in the self.imnorm class
        variable

        Returns
        -------

        psf: `xp.ndarray`
            Normalized (optional) wavefront array.

        '''

        fosys = self.init_fosys()
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        if self.source_offset[0]>0 or self.source_offset[1]>0:
            inwave.tilt(Xangle=self.source_offset[0]*self.as_per_lamD, Yangle=self.source_offset[1]*self.as_per_lamD)
        
        _, wfs = fosys.calc_psf(inwave=inwave, normalize='none', return_final=True, return_intermediates=False)
        wf = utils.rotate_arr(wfs[-1].wavefront, -self.det_rotation) if abs(self.det_rotation)>0 else wfs[-1].wavefront
        wf /= np.sqrt(self.Imax_ref) # Normalize by the unocculted PSF peak
        if pixelscale:
            return wf, wfs[-1].pixelscale
        else:
            return wf
    
    def snap(self,):
        '''
        Calculate the and return the PSF intensity.

        Parameters
        ----------

        plot: `bool`
            Display a plot of the PSF?
        
        Returns
        -------

        im: `xp.ndarray`
            Intensity of the final optical plane.
        '''
        
        return xp.abs(self.calc_wf())**2

    
    
