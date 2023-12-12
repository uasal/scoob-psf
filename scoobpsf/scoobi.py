import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import time
import copy
import os
from pathlib import Path

from .math_module import xp, _scipy, ensure_np_array
from . import imshows
from .utils import pad_or_crop
import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

import poppy


try:
    import scoobpy
    from scoobpy import utils
    scoobpy_avail = True
    print('this worked')
    
    from purepyindi import INDIClient, SwitchState
    client = INDIClient('localhost', 7624)
    client.start()
    print('Succesfully initialized testbed interface.')
except ImportError:
    print('Could not import scoobpy. Testbed interface unavailable.')
    scoobpy_avail = False

def set_texp(texp): # this one can also be done from the class, recommended to do that way so class attribute updates
        client.wait_for_properties({'scicam.exptime'}, timeout=10)
        client['scicam.exptime.target'] = texp
        time.sleep(0.5)

def set_roi(xc, yc, npix):
    # update roi parameters
    client.wait_for_properties(['scicam.roi_region_x', 'scicam.roi_region_y', 'scicam.roi_region_h' ,'scicam.roi_region_w', 'scicam.roi_set'])
    client['scicam.roi_region_x.target'] = xc
    client['scicam.roi_region_y.target'] = yc
    client['scicam.roi_region_h.target'] = npix
    client['scicam.roi_region_w.target'] = npix
    time.sleep(1)
    client['scicam.roi_set.request'] = SwitchState.ON
    time.sleep(1)

def move_psf(x_pos, y_pos):
    client.wait_for_properties(['stagepiezo.stagepupil_x_pos', 'stagepiezo.stagepupil_y_pos'])
    utils.move_relative(client, 'stagepiezo.stagepupil_x_pos', x_pos)
    time.sleep(0.25)
    utils.move_relative(client, 'stagepiezo.stagepupil_y_pos', y_pos)
    time.sleep(0.25)
    
def move_pol(rel_pos): # this will change the throughput by rotating the polarizer
    # FIXME
    utils.move_relative(client, 'stagepiezo.stagepol', rel_pos)
    
def move_fold(pos):
    # FIXME
    client['stagepiezo.stagefold_pos.current'] = pos

# define more functions for moving the fold mirror, using the tip tilt mirror, and the polarizers

class SCOOBI():

    def __init__(self, 
                dm_channel,
                dm_ref=np.zeros((34,34)),
                 dm_delay=0.01,
                x_shift=0,
                y_shift=0,
                nims=1,
                 npsf=256,
                 normalize=False,
                 texp_ref=None,
                 max_ref=None,
                ):
        
        # FIXME: make the science camera such that it can be specified

        # FIXME: add functionality to automatically subtract dark frames from newly captured frames

        # FIXME: add functionality for fiber attenuator, include for normalization

        self.is_model = False
        
        self.wavelength_c = 633e-9*u.m
        
        # Init all DM settings
        self.Nact = 34
        self.Nacts = 952
        self.dm_shape = (self.Nact,self.Nact)
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        self.full_stroke = 1.5e-6*u.m
        
        self.delay = 0.5
        self.dmctot = utils.connect_to_dmshmim(channel='dm00disp') # the total shared memory image
        self.dm_channel = dm_channel
        self.dmc = utils.connect_to_dmshmim(channel=dm_channel) # channel used for writing to DM
        
        self.dm_ref = dm_ref
        self.dm_delay = dm_delay
        self.dm_gain = 1
        
        bad_acts = [(21,25)]
        self.bad_acts = []
        for act in bad_acts:
            self.bad_acts.append(act[1]*self.Nact + act[0])
            
        self.dm_mask = np.ones((self.Nact,self.Nact))
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to_value(u.mm)*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>10.5] = 0
        
        self.dm_zernikes = poppy.zernike.arbitrary_basis(self.dm_mask, nterms=15, outside=0)

        # Init camera settings
        self.cam = utils.connect_to_camshmim()
        self.npsf = 64
        self.psf_pixelscale = 4.63e-6*u.m/u.pix
        self.psf_pixelscale_lamD = (1/5) * self.psf_pixelscale.to(u.m/u.pix).value/4.63e-6
        self.nims = nims
        
        client.wait_for_properties({'scicam.exptime'}, timeout=10)
        self._texp = client['scicam.exptime.current']
        
        self.texp_ref = texp_ref
        self.max_ref = max_ref
        
        self.npsf = npsf
        
        self.normalize = normalize
        if self.texp_ref is not None and self.max_ref is not None:
            self.imnorm = self.max0 / self.texp0
        
        self.x_shift = x_shift
        self.y_shift = y_shift

        self.use_cupy = False
    
    
    @property
    def texp(self):
        return self._texp

    @texp.setter
    def texp(self, value):
        client.wait_for_properties({'scicam.exptime'}, timeout=10)
        client['scicam.exptime.target'] = value
        time.sleep(0.5)
        self._texp = client['scicam.exptime.current']
        
    @property
    def emgain(self):
        return self._emgain

    @emgain.setter
    def emgain(self, value):
        client.wait_for_properties({'scicam.emgain'}, timeout=10)
        client['scicam.emgain.target'] = value
        time.sleep(0.5)
        self._emgain = client['scicam.emgain.current']
    # make something for emgain, client['scicam.emgain.target'] = 20
    
    def zero_dm(self):
        self.dmc.write(np.zeros(self.dm_shape))
        time.sleep(self.dm_delay)
    
    def reset_dm(self):
        self.dmc.write(self.dm_ref)
        time.sleep(self.dm_delay)
    
    def set_dm(self, dm_command):
        self.dmc.write(dm_command*1e6)
        time.sleep(self.dm_delay)
    
    def add_dm(self, dm_command):
        dm_state = self.get_dm()
        self.dmc.write( (dm_state + dm_command)*1e6 )
        time.sleep(self.dm_delay)
               
    def get_dm(self, total=False):
        if total:
            return self.dmctot.grab_latest()/1e6
        else:
            return self.dmc.grab_latest()/1e6
    
    def show_dm(self):
        imshows.imshow2(self.get_dm(), self.get_dm(total=True), self.dm_channel, 'dm00disp')
    
    def close_dm(self):
        self.dmc.close()
        
    def snap(self, nims=None, plot=False, vmin=None):
        
        Nims = self.nims if nims is None else nims
        
        if self.nims>1:
            ims = self.cam.grab_many(Nims)
            im = np.sum(ims, axis=0)/Nims
        else:
            im = self.cam.grab_latest()

        if self.use_cupy:
            im = xp.array(im)
        
        im = _scipy.ndimage.shift(im, (self.y_shift, self.x_shift), order=0)
        im = pad_or_crop(im, self.npsf)
        if self.normalize and self.texp_ref is not None and self.Imax_ref is not None:
            im *= (1/self.Imax_ref) * (self.texp_ref/self.texp)
            
        if plot:
            imshows.imshow1(im, lognorm=True, pxscl=self.psf_pixelscale_lamD, grid=True, vmin=vmin)
        
        return im
    
    
        
        