import numpy as np

import poppy
if poppy.accel_math._USE_CUPY:
    import cupy as cp
    import cupyx.scipy
    xp = cp
    _scipy = cupyx.scipy
else:
    xp = np
    _scipy = scipy
    
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import copy

from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image

import os
from pathlib import Path

import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

class SCOOBM():

    def __init__(self, 
                 wavelength=None, 
                 npix=128, 
                 oversample=2048/128,
                 npsf=400,
                 psf_pixelscale=4.63e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None,
                 norm='first',
                 imnorm=1,
                 det_rotation=0,
                 offset=(0,0),
                 use_opds=False,
                 use_aps=False,
                 inf_fun=None,
                 OPD=None,
                 RETRIEVED=None,
                 FPM=None,
                 LYOT=None):
        
        poppy.accel_math.update_math_settings()
        
        self.is_model = True
        
        self.pupil_diam = 6.75*u.mm
        self.wavelength_c = 632.8e-9*u.m
        
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.npix = npix
        self.oversample = oversample
        self.n = int(self.npix*self.oversample)
        
        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = (1/(5)) * self.psf_pixelscale.to(u.m/u.pix).value/4.63e-6
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 4.63e-6*u.m/u.pix / self.psf_pixelscale_lamD/(1/5)
        
        self.det_rotation = det_rotation
        self.norm = norm # specifies input wavefront normalization, see POPPY for more details
        self.imnorm = imnorm # image normalization factor
        
        self.offset = offset
        self.use_opds = use_opds
        self.use_aps = use_aps
        
        self.OPD = poppy.ScalarTransmission(name='OPD Place-holder') if OPD is None else OPD
        self.RETRIEVED = poppy.ScalarTransmission(name='Phase Retrieval Place-holder') if RETRIEVED is None else RETRIEVED
        self.FPM = poppy.ScalarTransmission(name='FPM Place-holder') if FPM is None else FPM
        self.LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if LYOT is None else LYOT
        
        self.inf_fun = str(module_path/'inf.fits') if inf_fun is None else inf_fun
        self.init_dm()
        self.init_opds()
        
    def getattr(self, attr):
        return getattr(self, attr)

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
        
        self.dm_zernikes = poppy.zernike.arbitrary_basis(cp.array(self.dm_mask), nterms=15, outside=0).get()
        
        bad_acts = [(21,25)]
        self.bad_acts = []
        for act in bad_acts:
            self.bad_acts.append(act[1]*self.Nact + act[0])

        self.DM = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM', 
                                                   actuator_spacing=self.act_spacing, 
                                                   influence_func=self.inf_fun,
                                                  )
        
    def reset_dm(self):
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm(self, dm_command):
        self.DM.set_surface(dm_command)
        
    def add_dm(self, dm_command):
        self.DM.set_surface(self.get_dm() + dm_command)
        
    def get_dm(self):
        return self.DM.surface.get()
    
    def show_dm(self):
        wf = poppy.FresnelWavefront(beam_radius=self.dm_active_diam/2, npix=self.npix, oversample=1)
        dm_command = self.get_dm()
        dm_surface = self.DM.get_opd(wf).get() if poppy.accel_math._USE_CUPY else self.DM.get_opd(wf)
        surf_ext = wf.pixelscale.to_value(u.mm/u.pix)*self.npix/2
        
        fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), dpi=125)
        im = ax[0].imshow(dm_command)
        ax[0].set_title('DM Command')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="4%", pad=0.075)
        fig.colorbar(im, cax=cax)
        
        im = ax[1].imshow(dm_surface, extent=[-surf_ext, surf_ext, -surf_ext, surf_ext])
        ax[1].set_title('DM Surface')
        ax[1].set_xlabel('mm')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="4%", pad=0.075)
        fig.colorbar(im, cax=cax)
        
        plt.subplots_adjust(wspace=0.3)
    
    def init_inwave(self):
        self.pupil_diam = 6.8*u.mm
        
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        self.inwave = inwave
        
    def init_opds(self):
        opd_dir = Path(os.path.dirname(str(module_path)))/'scoob-opds'
        
        self.m3_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'M3.fits'), opdunits='meters', planetype=inter)
        self.oap1_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP1.fits'), opdunits='meters', planetype=inter)
        self.oap2_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP2.fits'), opdunits='meters', planetype=inter)
        self.oap3_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'OAP3.fits'), opdunits='meters', planetype=inter)
        self.flat1_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'FLAT1.fits'), opdunits='meters', planetype=inter)
        self.flat2_opd = poppy.FITSOpticalElement(opd=str(opd_dir/'FLAT2.fits'), opdunits='meters', planetype=inter)
    
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
        
        OPD = poppy.ScalarTransmission() if self.OPD is None else self.OPD
        RETRIEVED = poppy.ScalarTransmission() if self.RETRIEVED is None else self.RETRIEVED
        FPM = poppy.ScalarTransmission() if self.FPM is None else self.FPM
        LYOT = poppy.ScalarTransmission() if self.LYOT is None else self.LYOT
        
        d_pupil_stop_flat = 42.522683120520355*u.mm
        d_flat_oap1 = 89.33864507546247*u.mm
        d_oap1_oap2 = 307.2973451416505*u.mm
        d_oap2_DM = 168.84723167004802*u.mm + 0*u.mm
        d_DM_oap3 = 460.43684751808945*u.mm
#         d_DM_oap3 = 462.6680664924039*u.mm
        d_oap3_FPM = 462.6680664924039*u.mm
        d_FPM_flat2 = 144.6375029385836*u.mm 
        d_flat2_lens = 200*u.mm - 144.6375029385836*u.mm
        d_lens_LYOT = 200*u.mm
#         d_LYOT_scicam = 75*u.mm
#         d_scicam_image = 75*u.mm
        d_LYOT_scicam = 150*u.mm
        d_scicam_image = 150*u.mm

        fl_oap0 = self.oaefl(293.6,27)*u.mm
        fl_oap1 = self.oaefl(254,40)*u.mm
        fl_oap2 = self.oaefl(346,55)*u.mm
        fl_oap3 = self.oaefl(914.4,100)*u.mm
#         print(fl_oap0, fl_oap1, fl_oap2, fl_oap3)

        fl_lens = 200*u.mm
#         fl_scicam_lens = 75*u.mm
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
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        
        fosys.add_optic(pupil_stop)
        if self.use_opds: fosys.add_optic(self.flat1_opd)
        fosys.add_optic(OPD)
        
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
        
#         fosys.add_optic(poppy.ScalarTransmission(), distance=d_scicam_image)
        fosys.add_detector(pixelscale=self.psf_pixelscale.to(u.m/u.pix), fov_pixels=self.npsf, distance=d_scicam_image)
        
        self.fosys = fosys
    
    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        _, wfs = self.fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_intermediates=True)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        if abs(self.det_rotation)>0:
            rotated_wf = self.rotate_wf(wfs[-1])
            wfs[-1] = rotated_wf
        wfs[-1].wavefront /= np.sqrt(self.imnorm)
        
        return wfs
    
    def calc_psf(self, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        _, wfs = self.fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_final=True, return_intermediates=False)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        wf = self.rotate_wf(wfs[-1]) if abs(self.det_rotation)>0 else wfs[-1]

        return wf.wavefront/np.sqrt(self.imnorm)
    
    def snap(self): # method for getting the PSF in photons
        self.init_fosys()
        self.init_inwave()
        
        _, wfs = self.fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_intermediates=False, return_final=True)
        
        wf = self.rotate_wf(wfs[-1]) if abs(self.det_rotation)>0 else wfs[-1]
        
        return wf.intensity/self.imnorm
    
    def rotate_wf(self, wave):
        wavefront = wave.wavefront
        wavefront_r = _scipy.ndimage.rotate(xp.real(wavefront), angle=-self.det_rotation, reshape=False, order=1)
        wavefront_i = _scipy.ndimage.rotate(xp.imag(wavefront), angle=-self.det_rotation, reshape=False, order=1)
        
        wave.wavefront = (wavefront_r + 1j*wavefront_i)
        return wave
    
