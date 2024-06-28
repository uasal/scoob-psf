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
    from scoobpy import utils as scoob_utils
    from magpyx.utils import ImageStream

    import purepyindi
    from purepyindi import INDIClient
    client0 = INDIClient('localhost', 7624)
    client0.start()

    import purepyindi2
    from purepyindi2 import IndiClient
    client = IndiClient()
    client.connect()
    client.get_properties()

    scoobpy_avail = True
    print('Succesfully initialized testbed interface.')
except ImportError:
    print('Could not import scoobpy. Testbed interface unavailable.')
    scoobpy_avail = False

def move_psf(x_pos, y_pos):
        client0.wait_for_properties(['stagepiezo.stagepupil_x_pos', 'stagepiezo.stagepupil_y_pos'])
        scoob_utils.move_relative(client0, 'stagepiezo.stagepupil_x_pos', x_pos)
        time.sleep(0.25)
        scoob_utils.move_relative(client0, 'stagepiezo.stagepupil_y_pos', y_pos)
        time.sleep(0.25)

def set_zwo_roi(xc, yc, npix):
    # update roi parameters
    client0.wait_for_properties(['scicam.roi_region_x', 'scicam.roi_region_y', 'scicam.roi_region_h' ,'scicam.roi_region_w', 'scicam.roi_set'])
    client0['scicam.roi_region_x.target'] = xc
    client0['scicam.roi_region_y.target'] = yc
    client0['scicam.roi_region_h.target'] = npix
    client0['scicam.roi_region_w.target'] = npix
    time.sleep(1)
    client0['scicam.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(1)
    
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
                 cam_channel,
                 dm_ref=np.zeros((34,34)),
                 x_shift=0,
                 y_shift=0,
                 Nframes=1,
                 npsf=256,
                 exp_time=0.001*u.s,
                 gain=1, 
                 attenuation=5,
                 normalize=True, 
                 Imax_ref=None,
                ):
        
        # FIXME: make the science camera such that it can be specified

        # FIXME: add functionality to automatically subtract dark frames from newly captured frames

        self.is_model = False
        
        self.wavelength_c = 633e-9*u.m
        
        self.CAM = ImageStream(cam_channel)

        self.dm_channel = dm_channel
        self.DM = scoob_utils.connect_to_dmshmim(channel=dm_channel) # channel used for writing to DM
        self.DMT = scoob_utils.connect_to_dmshmim(channel='dm00disp') # the total shared memory image

        # Init all DM settings
        self.Nact = 34
        self.Nacts = 952
        self.dm_shape = (self.Nact,self.Nact)
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        self.full_stroke = 1.5e-6*u.m
        
        self.dm_ref = dm_ref
        self.dm_gain = 1
        self.reset_dm()
        
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
        self.psf_pixelscale = 3.76e-6*u.m/u.pix
        self.psf_pixelscale_lamD = 0.2711864406779661
        self.Nframes = Nframes
        
        self.npsf = npsf
        self.nbits = 16

        self.blacklevel = 63
        self.gain = gain
        self.exp_time = exp_time
        self.attenuation = attenuation

        self.att_ref = None

        self.normalize = normalize
        self.Imax_ref = Imax_ref
        
        self.x_shift = x_shift
        self.y_shift = y_shift

        self.subtract_bias = False

        self.exp_times = None
        self.Nframes_per_exp = None

        self.exp_time_delay = 0.1
        self.attenuation_delay = 0.1
        self.dm_delay = 0.01
    
    # @property
    # def bias(self):
    #     return self._bias

    # @bias.setter
    # def bias(self, value):
    #     client['nsv571.blacklevel.blacklevel'] = value
    #     time.sleep(0.5)
    #     self._bias = value

    @property
    def exp_time(self):
        return self._exp_time

    @exp_time.setter
    def exp_time(self, value):
        client.get_properties()
        # client['nsv571.exptime.exptime'] = value.to_value(u.s)
        # time.sleep(0.5)
        # self._exp_time = value
        client0.wait_for_properties(['scicam_2600.exptime'])
        client0['scicam_2600.exptime.target'] = value.to_value(u.s)
        time.sleep(self.exp_time_delay)
        self._exp_time = client0['scicam_2600.exptime.current'] * u.s 
        
    # @property
    # def gain(self):
    #     return self._gain

    # @gain.setter
    # def gain(self, value):
    #     if value>10 or value<1:
    #         raise ValueError('Gain value cannot be greater than 10 or less than 1.')
    #     client['nsv571.gain.gain']= value
    #     time.sleep(0.5)
    #     self._gain = value

    @property
    def attenuation(self):
        return self._attenuation
    
    @attenuation.setter
    def attenuation(self, value):
        client['fiberatten.atten.target'] = value
        time.sleep(self.attenuation_delay)
        self._attenuation = value

    def zero_dm(self):
        self.DM.write(np.zeros(self.dm_shape))
        time.sleep(self.dm_delay)
    
    def reset_dm(self):
        self.DM.write(ensure_np_array(self.dm_ref))
        time.sleep(self.dm_delay)
    
    def set_dm(self, dm_command):
        self.DM.write(ensure_np_array(dm_command)*1e6)
        time.sleep(self.dm_delay)
    
    def add_dm(self, dm_command):
        dm_state = ensure_np_array(self.get_dm())
        self.DM.write( 1e6*(dm_state + ensure_np_array(dm_command)) )
        time.sleep(self.dm_delay)
               
    def get_dm(self, total=False):
        if total:
            return xp.array(self.DMT.grab_latest())/1e6
        else:
            return xp.array(self.DM.grab_latest())/1e6
    
    def show_dm(self):
        imshows.imshow2(self.get_dm(), self.get_dm(total=True), self.dm_channel, 'dm00disp')
    
    def close_dm(self):
        self.DM.close()

    def snap(self, nims=None, plot=False, vmin=None):
        
        if self.exp_times is not None and self.Nframes_per_exp is not None:
            return self.snap_many(plot=True)
        
        if self.Nframes>1:
            ims = self.CAM.grab_many(self.Nframes)
            im = np.sum(ims, axis=0)/self.Nframes
        else:
            im = self.CAM.grab_latest()
        
        im = xp.array(im)
        im = _scipy.ndimage.shift(im, (self.y_shift, self.x_shift), order=0)
        im = pad_or_crop(im, self.npsf)

        if self.subtract_bias:
            im -= self.bias
            
        if self.normalize:
            im *= (1/self.gain) * 10**(self.attenuation/10) * (1/self.exp_time.to_value(u.s))
            if self.Imax_ref is not None:
                im /= self.Imax_ref
            
        if plot:
            imshows.imshow1(im, lognorm=True, pxscl=self.psf_pixelscale_lamD, grid=True, vmin=vmin)
        
        return im
    
    def snap_many(self, plot=False):
        if len(self.exp_times)!=len(self.Nframes_per_exp):
            raise ValueError('The specified number of frames per exposure time must match the specified number of exposure times.')
            
        total_im = 0.0
        pixel_weights = 0.0
        for i in range(len(self.exp_times)):
            self.exp_time = self.exp_times[i]
            self.Nframes = self.Nframes_per_exp[i]

            frames = self.CAM.grab_many(self.Nframes)
            mean_frame = np.sum(frames, axis=0)/self.Nframes
            mean_frame = _scipy.ndimage.shift(mean_frame, (self.y_shift, self.x_shift), order=0)
            mean_frame = pad_or_crop(mean_frame, self.npsf)
                
            pixel_sat_mask = mean_frame > self.sat_thresh

            if self.subtract_bias is not None:
                mean_frame -= self.subtract_bias
            
            pixel_weights += ~pixel_sat_mask
            normalized_im = mean_frame/self.exp_time 
            normalized_im[pixel_sat_mask] = 0 # mask out the saturated pixels

            if plot: 
                imshows.imshow3(pixel_weights, mean_frame, normalized_im, 
                                'Pixel Weight Map', 
                                f'Frame:\nExposure Time = {self.exp_time:.2e}s', 
                                'Masked Flux Image', 
                                # lognorm2=True, lognorm3=True,
                                )
                
            total_im += normalized_im
            
        total_im /= pixel_weights

        return total_im
    
    # def snap_llowfsc(self):

    
        
        