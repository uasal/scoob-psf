import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import time
import copy
import os
from pathlib import Path

from .math_module import xp, _scipy, ensure_np_array
from .utils import pad_or_crop
import scoobpsf

module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

import poppy

from scoobpy import utils as scoob_utils
import purepyindi
import purepyindi2
from magpyx.utils import ImageStream

def move_psf(x_pos, y_pos, client):
    client.wait_for_properties(['stagepiezo.stagepupil_x_pos', 'stagepiezo.stagepupil_y_pos'])
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_x_pos', x_pos)
    time.sleep(0.25)
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_y_pos', y_pos)
    time.sleep(0.25)

def set_zwo_roi(xc, yc, npix, client, delay=0.25):
    # update roi parameters
    client.wait_for_properties(['scicam.roi_region_x', 'scicam.roi_region_y', 'scicam.roi_region_h' ,'scicam.roi_region_w', 'scicam.roi_set'])
    client['scicam.roi_region_x.target'] = xc
    client['scicam.roi_region_y.target'] = yc
    client['scicam.roi_region_h.target'] = npix
    client['scicam.roi_region_w.target'] = npix
    time.sleep(delay)
    client['scicam.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_zwo_exp_time(exp_time, client, delay=0.1):
    client.wait_for_properties(['scicam.exptime'])
    client['scicam.exptime.target'] = exp_time
    time.sleep(delay)

def get_zwo_exp_time(client):
    client.wait_for_properties(['scicam.exptime'])
    return client['scicam.exptime.current']

def set_zwo_emgain(gain, client, delay=0.1):
    client.wait_for_properties(['scicam.emgain'])
    client['scicam.emgain.target'] = gain
    time.sleep(delay)

def get_zwo_emgain(client):
    client.wait_for_properties(['scicam.emgain'])
    return client['scicam.emgain.current']

def set_fib_atten(value, client, delay=0.1):
    client['fiberatten.atten.target'] = value
    time.sleep(delay)

def get_fib_atten(client):
    return client['fiberatten.atten.current']

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
        # I need to subrtract dark frames 
        # and account for negative pixels in the dark subtracted frames

        self.wavelength_c = 633e-9*u.m
        
        self.CAM = ImageStream(cam_channel)

        self.dm_channel = dm_channel
        self.DM = scoob_utils.connect_to_dmshmim(channel=dm_channel) # channel used for writing to DM
        self.DMT = scoob_utils.connect_to_dmshmim(channel='dm00disp') # the total shared memory image

        self.dm_delay = 0.1

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
        self.psf_pixelscale_lamD = 0.307
        self.Nframes = Nframes
        
        self.npsf = npsf
        self.nbits = 16

        self.texp = 1
        self.att = 1
        self.gain = 1

        self.df = None
        self.subtract_dark = False

        self.return_ni = False

        self.Imax_ref = 1
        self.texp_ref = 1
        self.att_ref = 1
        self.gain_ref = 1
        
        self.x_shift = x_shift
        self.y_shift = y_shift

    def set_zwo_exp_time(self, exp_time, client, delay=0.25):
        client.wait_for_properties(['scicam.exptime'])
        client['scicam.exptime.target'] = exp_time
        time.sleep(delay)
        self.texp = client['scicam.exptime.current']
        print(f'Set the ZWO exposure time to {self.texp:.2e}s')

    def set_fib_atten(self, value, client, delay=0.1):
        client['fiberatten.atten.target'] = value
        time.sleep(delay)
        self.att = value
        print(f'Set the fiber attenuation to {value:.1f}')

    def set_zwo_emgain(self, gain, client, delay=0.1):
        client.wait_for_properties(['scicam.emgain'])
        client['scicam.emgain.target'] = gain
        time.sleep(delay)
        self.gain = gain
        print(f'Set the ZWO gain setting to {gain:.1f}')

    def take_dark(self, client, delay=0.5):
        client.wait_for_properties(['fiberatten.atten'])
        client['stagelinear.presetName.block_in'] = purepyindi.SwitchState.ON
        time.sleep(delay)
        self.df = self.snap()
        client.wait_for_properties(['fiberatten.atten'])
        client['stagelinear.presetName.block_out'] = purepyindi.SwitchState.ON
        time.sleep(delay)

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
    
    def close_dm(self):
        self.DM.close()

    def normalize(self, image):
        image_ni = image/self.Imax_ref
        image_ni *= (self.texp_ref/self.texp)
        image_ni *= 10**((self.att-self.att_ref)/10)
        image_ni *= 10**(-self.gain/20 * 0.1) / 10**(-self.gain_ref/20 * 0.1)
        # gain ~ 10^(-gain_setting/20 * 0.1)
        return image_ni

    def snap(self, normalize=False, plot=False, vmin=None):
        if self.Nframes>1:
            ims = self.CAM.grab_many(self.Nframes)
            im = np.sum(ims, axis=0)/self.Nframes
        else:
            im = self.CAM.grab_latest()
        
        im = xp.array(im)
        im = _scipy.ndimage.shift(im, (self.y_shift, self.x_shift), order=0)
        im = pad_or_crop(im, self.npsf)

        if self.subtract_dark and self.df is not None:
            im -= self.df
            
        if self.return_ni:
            im = self.normalize(im)
        
        return im
    
    def snap_many(self,):
        return self.CAM.grab_many(self.Nframes)
    
def snap_many(images, Nframes_per_exp, exp_times, gains, plot=False):
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

    
        
        