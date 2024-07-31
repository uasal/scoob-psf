from .math_module import xp, _scipy, ensure_np_array
from .imshows import imshow1, imshow2, imshow3

import numpy as np
import scipy
from astropy.io import fits
import pickle
from matplotlib.patches import Circle, Rectangle

def make_grid(npix, pixelscale=1, half_shift=False):
    if half_shift:
        y,x = (xp.indices((npix, npix)) - npix//2 + 1/2)*pixelscale
    else:
        y,x = (xp.indices((npix, npix)) - npix//2)*pixelscale
    return x,y

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def rotate_arr(arr, rotation, reshape=False, order=3):
    if arr.dtype == complex:
        arr_r = _scipy.ndimage.rotate(xp.real(arr), angle=rotation, reshape=reshape, order=order)
        arr_i = _scipy.ndimage.rotate(xp.imag(arr), angle=rotation, reshape=reshape, order=order)
        
        rotated_arr = arr_r + 1j*arr_i
    else:
        rotated_arr = _scipy.ndimage.rotate(arr, angle=rotation, reshape=reshape, order=order)
    return rotated_arr

def interp_arr(arr, pixelscale, new_pixelscale, order=3):
        Nold = arr.shape[0]
        old_xmax = pixelscale * Nold/2

        x,y = xp.ogrid[-old_xmax:old_xmax-pixelscale:Nold*1j,
                       -old_xmax:old_xmax-pixelscale:Nold*1j]

        Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) - 1
        new_xmax = new_pixelscale * Nnew/2

        newx,newy = xp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                             -new_xmax:new_xmax-new_pixelscale:Nnew*1j]

        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0] - x0
        dy = y[0,1] - y0

        ivals = (newx - x0)/dx
        jvals = (newy - y0)/dy

        coords = xp.array([ivals, jvals])

        interped_arr = _scipy.ndimage.map_coordinates(arr, coords, order=order)
        return interped_arr

import poppy
from .import imshows
from skimage.filters import threshold_otsu
import astropy.units as u
def process_pr_data(pr_amp, pr_phs, npup, pr_rotation, 
                    pixelscale=None,
                    N=None,
                    amp_norm=1,
                    remove_modes=None,
                    ):
    
    if isinstance(pr_amp, str):
        pr_amp = xp.array(fits.getdata(pr_amp))/amp_norm
    if isinstance(pr_phs, str):
        pr_phs = xp.array(fits.getdata(pr_phs))
    # imshows.imshow2(pr_amp, pr_phs)

    pr_amp = pad_or_crop(pr_amp, npup)
    pr_phs = pad_or_crop(pr_phs, npup)
    imshows.imshow2(pr_amp, pr_phs)

    if remove_modes is not None:
        thresh_mask = xp.array(pr_amp>threshold_otsu(pr_amp))
        pr_phs[~thresh_mask] = xp.NaN
        imshows.imshow2(thresh_mask, pr_phs)

        Zs = poppy.zernike.arbitrary_basis(thresh_mask, nterms=remove_modes, outside=0)
        imshows.imshow3(Zs[0], Zs[1], Zs[2])
        
        Zc = lstsq(Zs, pr_phs)

        for i in range(remove_modes):
            pr_phs -= Zc[i] * Zs[i]
        pr_amp[~thresh_mask] = 0.0
        pr_phs[~thresh_mask] = 0.0
    imshows.imshow2(pr_amp, pr_phs)

    pr_amp = _scipy.ndimage.rotate(pr_amp, angle=pr_rotation, reshape=False, order=3)
    pr_phs = _scipy.ndimage.rotate(pr_phs, angle=pr_rotation, reshape=False, order=3)

    if pixelscale is not None:
        pr_amp = interp_arr(pr_amp, (6.75*u.mm/(npup*u.pix)).to_value(u.m/u.pix), pixelscale.to_value(u.m/u.pix))
        pr_phs = interp_arr(pr_phs, (6.75*u.mm/(npup*u.pix)).to_value(u.m/u.pix), pixelscale.to_value(u.m/u.pix))
    imshows.imshow2(pr_amp/2, pr_phs, 'Pupil Plane Amplitude', 'Pupil Plane Phase', 
                    pxscl=6.75*u.mm/(npup*u.pix), xlabel='X [mm]')

    wfe = pr_amp*xp.exp(1j*pr_phs)

    if N is not None:
        pad_or_crop(wfe, N)

    return wfe

def lstsq(modes, data):
    """Least-Squares fit of modes to data.

    Parameters
    ----------
    modes : iterable
        modes to fit; sequence of ndarray of shape (m, n)
    data : numpy.ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    numpy.ndarray
        fit coefficients

    """
    mask = xp.isfinite(data)
    data = data[mask]
    modes = xp.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    modes = modes[:, mask.ravel()].T  # transpose moves modes to columns, as needed for least squares fit
    c, *_ = xp.linalg.lstsq(modes, data, rcond=None)
    return c


def save_fits(fpath, data, header=None, ow=True, quiet=False):
    if header is not None:
        keys = list(header.keys())
        hdr = fits.Header()
        for i in range(len(header)):
            hdr[keys[i]] = header[keys[i]]
    else: 
        hdr = None
    
    data = ensure_np_array(data)
    
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(str(fpath), overwrite=ow) 
    if not quiet: print('Saved data to: ', str(fpath))

# functions for saving python objects
def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data


def generate_wfe(diam, 
                 npix=256, oversample=1, 
                 wavelength=500*u.nm,
                 opd_index=2.5, amp_index=2, 
                 opd_seed=1234, amp_seed=12345,
                 opd_rms=10*u.nm, amp_rms=0.05,
                 remove_modes=3, # defaults to removing piston, tip, and tilt
                 ):
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_opd = poppy.StatisticalPSDWFE(index=opd_index, wfe=opd_rms, radius=diam/2, seed=opd_seed).get_opd(wf)
    wfe_amp = poppy.StatisticalPSDWFE(index=amp_index, wfe=amp_rms*u.nm, radius=diam/2, seed=amp_seed).get_opd(wf)
    
    wfe_amp = xp.asarray(wfe_amp)
    wfe_opd = xp.asarray(wfe_opd)

    mask = poppy.CircularAperture(radius=diam/2).get_transmission(wf)>0
    Zs = poppy.zernike.arbitrary_basis(mask, nterms=remove_modes, outside=0)
    
    Zc_amp = lstsq(Zs, wfe_amp)
    Zc_opd = lstsq(Zs, wfe_opd)
    for i in range(3):
        wfe_amp -= Zc_amp[i] * Zs[i]
        wfe_opd -= Zc_opd[i] * Zs[i]

    mask = poppy.CircularAperture(radius=diam/2).get_transmission(wf)>0
    wfe_rms = xp.sqrt(xp.mean(xp.square(wfe_opd[mask])))
    wfe_opd *= opd_rms.to_value(u.m)/wfe_rms

    wfe_amp = wfe_amp*1e9 + 1

    wfe_amp_rms = xp.sqrt(xp.mean(xp.square(wfe_amp[mask]-1)))
    wfe_amp *= amp_rms/wfe_amp_rms

    wfe = wfe_amp * xp.exp(1j*2*np.pi/wavelength.to_value(u.m) * wfe_opd)
    wfe *= poppy.CircularAperture(radius=diam/2).get_transmission(wf)

    return wfe, mask

def centroid(arr, rounded=False):
    weighted_sum_x = 0
    total_sum_x = 0
    for i in range(arr.shape[1]):
        weighted_sum_x += np.sum(arr[:,i])*i
        total_sum_x += np.sum(arr[:,i])
    xc = round(weighted_sum_x/total_sum_x) if rounded else weighted_sum_x/total_sum_x
    
    weighted_sum_y = 0
    total_sum_y = 0
    for i in range(arr.shape[0]):
        weighted_sum_y += np.sum(arr[i,:])*i
        total_sum_y += np.sum(arr[i,:])
        
    yc = round(weighted_sum_y/total_sum_y) if rounded else weighted_sum_y/total_sum_y
    return xp.array([xc, yc])


import skimage

def create_fourier_command(x_cpa=10, y_cpa=10, Nact=34):
    # cpa = cycles per aperture
    # max cpa must be Nact/2
    if x_cpa>Nact/2 or y_cpa>Nact/2:
        raise ValueError('The cycles per aperture is too high for the specified number of actuators.')
    y,x = xp.indices((Nact, Nact)) - Nact//2
    fourier_command = xp.cos(2*np.pi*(x_cpa*x + y_cpa*y)/Nact)
    return fourier_command

def measure_center_and_angle(waffle_im, psf_pixelscale_lamD, im_thresh=1e-4, r_thresh=12,
                           verbose=True, 
                           plot=True):
    npsf = waffle_im.shape[0]
    y,x = (xp.indices((npsf, npsf)) - npsf//2)*psf_pixelscale_lamD
    r = xp.sqrt(x**2 + y**2)
    waffle_mask = (waffle_im >im_thresh) * (r>r_thresh)

    centroids = []
    for i in [0,1]:
        for j in [0,1]:
            arr = waffle_im[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            mask = waffle_mask[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            cent = np.flip(skimage.measure.centroid(ensure_np_array(mask*arr)))
            cent[0] += i*npsf//2
            cent[1] += j*npsf//2
            centroids.append(cent)
            # print(cent)
            # imshow3(mask, arr, mask*arr, lognorm2=True,
            #         patches1=[Circle(cent, 1, fill=True, color='cyan')])
    centroids.append(centroids[0])
    centroids = np.array(centroids)
    centroids[[2,3]] = centroids[[3,2]]
    if verbose: print('Centroids:\n', centroids)

    if plot: 
        patches = []
        for i in range(4):
            patches.append(Circle(centroids[i], 1, fill=False, color='black'))
        imshow3(waffle_mask, waffle_im, waffle_mask*waffle_im, lognorm2=True, vmin2=1e-5,
                patches1=patches)

    mean_angle = 0.0
    for i in range(4):
        angle = np.arctan2(centroids[i+1][1] - centroids[i][1], centroids[i+1][0] - centroids[i][0]) * 180/np.pi
        if angle<0:
            angle += 360
        if 0<angle<90:
            angle = 90-angle
        elif 90<angle<180:
            angle = 180-angle
        elif 180<angle<270:
            angle = 270-angle
        elif 270<angle<360:
            angle = 360-angle
        mean_angle += angle/4
    if verbose: print('Angle: ', mean_angle)

    m1 = (centroids[0][1] - centroids[2][1])/(centroids[0][0] - centroids[2][0])
    m2 = (centroids[1][1] - centroids[3][1])/(centroids[1][0] - centroids[3][0])
    # print(m1,m2)
    b1 = -m1*centroids[0][0] + centroids[0][1]
    b2 =  -m2*centroids[1][0] + centroids[1][1]
    # print(b1,b2)

    # m1*x + b1 = m2*x + b2
    # (m1-m2) * x = b2 - b1
    xc = (b2 - b1) / (m1 - m2)
    yc = m1*xc + b1
    print('Measured center in X: ', xc)
    print('Measured center in Y: ', yc)

    xshift = np.round(npsf/2 - xc)
    yshift = np.round(npsf/2 - yc)
    print('Required shift in X: ', xshift)
    print('Required shift in Y: ', yshift)

    return xshift,yshift,mean_angle

def create_control_mask(npsf, pxscl, iwa=3, owa=12, edge=None, rotation=0, centered=True):
    if centered:
        x = xp.linspace(-npsf/2, npsf/2-1, npsf)*pxscl
    else:
        x = (xp.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*pxscl
    x,y = xp.meshgrid(x,x)
    r = xp.hypot(x, y)
    control_mask = (r < owa) * (r > iwa)
    if edge is not None: control_mask *= (x > edge)

    control_mask = _scipy.ndimage.rotate(control_mask, rotation, reshape=False, order=0)
    return control_mask
