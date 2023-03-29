import numpy as np
import cupy as cp
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import pickle
import time
import copy
import scipy

import poppy

try:
    import scoobpy
    from scoobpy import utils
    scoobpy_avail = True

    from purepyindi import INDIClient, SwitchState
    client = INDIClient('localhost', 7624)
    client.start()
except:
    print('Could not import scoobpy.')
    scoobpy_avail = False

def set_texp(texp):
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


class SCOOBI():

    def __init__(self, 
                dm_channel,
                dm_ref=None,
                x_shift=0,
                y_shift=0,
                nims=1,
                normalization=1):
        
        self.is_model = False
        
        self.wavelength_c = 633e-9*u.m

        self.Nact = 34
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
        
        self.dm_gain = 1
        
        bad_acts = [(21,25)]
        self.bad_acts = []
        for act in bad_acts:
            self.bad_acts.append(act[1]*self.Nact + act[0])
            
        self.dm_mask = np.ones((self.Nact,self.Nact))
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>10.5] = 0
        
        self.dm_zernikes = poppy.zernike.arbitrary_basis(self.dm_mask, nterms=15, outside=0)

        self.cam = utils.connect_to_camshmim()
        self.npsf = 64
        self.psf_pixelscale = 4.63e-6*u.m/u.pix
        self.psf_pixelscale_lamD = (1/5) * self.psf_pixelscale.to(u.m/u.pix).value/4.63e-6
        self.nims = nims

        client.wait_for_properties({'scicam.exptime'}, timeout=10)
        self.texp = client['scicam.exptime.current']
        self.normalization = 1

        self.x_shift = x_shift
        self.y_shift = y_shift



    @property
    def delay(self):
        return self.dm_delay
    
    @delay.setter
    def delay(self, delay):
        self.dm_delay = delay
        
    def reset_dm(self):
        self.dmc.write(np.zeros(self.dm_shape))
    
    def set_dm(self, dm_command):
        self.dmc.write(dm_command*1e6)
    
    def add_dm(self, dm_command):
        dm_state = self.get_dm()
        self.dmc.write( (dm_state + dm_command)*1e6 )
               
    def get_dm(self, total=False):
        if total:
            return self.dmctot.grab_latest()/1e6
        else:
            return self.dmc.grab_latest()/1e6
    
    def show_dm(self):
        misc.myimshow2(self.get_dm(), self.get_dm(total=True), self.dm_channel, 'dm00disp')
        
    def snap(self):
        if self.nims>1:
            ims = self.cam.grab_many(self.nims)
            im = np.sum(ims, axis=0)/self.nims
        else:
            im = self.cam.grab_latest()

        im = scipy.ndimage.shift(im, (self.y_shift, self.x_shift), order=0)
        
        im /= self.normalization

        return im
    
    
        
        
        