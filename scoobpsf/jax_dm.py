import numpy as np
import scipy
import astropy.units as u

from matplotlib import patches

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .math_module import ensure_np_array
from .utils import pad_or_crop
from .imshows import *

# MISCELLANEOUS FUNCTIONS
def ensure_np_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, jax.numpy.ndarray):
        return np.asarray(arr)
    elif isinstance(arr, cp.ndarray):
        return arr.get()

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2]
    else:
        arr_out = jnp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out = arr_out.at[x1:x2,x1:x2].set(arr_in)
    return arr_out

def shift(arr, shift, plot=False):
    arr = jnp.roll(arr, shift[1], axis=0)
    arr = jnp.roll(arr, shift[0], axis=1)

    if shift[1]>0:
        arr = arr.at[:shift[1], :].set(0)
    elif shift[1]<0:
        arr = arr.at[shift[1]:, :].set(0)

    if shift[0]>0:
        arr = arr.at[:, :shift[0]].set(0)
    elif shift[0]<0:
        arr = arr.at[:, shift[0]:].set(0)

    if plot: imshow1(ensure_np_array(arr))
    return arr

def interp_arr(arr, pixelscale, new_pixelscale, order=1):
        Nold = arr.shape[0]
        old_xmax = pixelscale * Nold/2

        x,y = jnp.ogrid[-old_xmax:old_xmax-pixelscale:Nold*1j,
                       -old_xmax:old_xmax-pixelscale:Nold*1j]

        Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) - 1
        new_xmax = new_pixelscale * Nnew/2

        newx,newy = jnp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                              -new_xmax:new_xmax-new_pixelscale:Nnew*1j]

        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0] - x0
        dy = y[0,1] - y0

        ivals = (newx - x0)/dx
        jvals = (newy - y0)/dy

        coords = jnp.array([ivals, jvals])

        interped_arr = jax.scipy.ndimage.map_coordinates(arr, coords, order=order)
        return interped_arr

def make_gaussian_inf_fun(act_spacing=300e-6*u.m, 
                          sampling=25, 
                          Nacts_per_inf=4, # number of influence functions across the grid
                          coupling=0.15,
                            plot=False,
                            save_fits=None):

    ng = int(sampling*Nacts_per_inf)

    pxscl = act_spacing/(sampling*u.pix)
    ext = Nacts_per_inf * act_spacing

    xs = (jnp.linspace(-ng/2,ng/2-1,ng)+1/2)*pxscl.value
    x,y = jnp.meshgrid(xs,xs)
    r = jnp.sqrt(x**2 + y**2)

    # d = act_spacing.value/1.25
    d = act_spacing.value/jnp.sqrt(-jnp.log(coupling))
    # print(d)

    inf = jnp.exp(-(r/d)**2)
    rcoupled = d*jnp.sqrt(-jnp.log(coupling)) 
    
    if plot:

        fig,ax = imshow1(ensure_np_array(inf), pxscl=pxscl, 
                         patches=[patches.Circle((0,0), rcoupled, fill=False, color='c'),
                                  patches.Circle((0,0), rcoupled/2, fill=False, color='g', linewidth=1.5)], 
                            display_fig=False, return_fig=True)
        if Nacts_per_inf%2==0:
            ticks = np.linspace(-ext.value/2, ext.value/2, Nacts_per_inf+1)
        else:
            ticks = np.linspace(-ext.value/2, ext.value/2, Nacts_per_inf)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid()
        display(fig)

    if save_fits is not None:
        hdr = fits.Header()
        hdr['SAMPLING'] = sampling
        hdr.comments['SAMPLING'] = '# pixels per actuator'
        hdr['NACTS'] = ext_act
        hdr.comments['NACTS'] = '# actuators across grid'
        inf_hdu = fits.PrimaryHDU(data=ensure_np_array(inf), header=hdr)
        inf_hdu.writeto(str(save_fits), overwrite=True)

    pixelscale = act_spacing/sampling/u.pix
    return inf, sampling, pixelscale

def make_dm_mask(Nact=34, plot=False):
    dm_mask = jnp.ones((Nact,Nact), dtype=bool)
    xx = jnp.linspace(0, Nact-1, Nact) - Nact/2 + 1/2
    # print(xx)
    x,y = jnp.meshgrid(xx,xx)
    r = jnp.sqrt(x**2 + y**2)
    inds = jnp.where(r>Nact//2+1/2)
    dm_mask = dm_mask.at[inds[0], inds[1]].set(0)
    if plot: imshow1(ensure_np_array(dm_mask))
    return dm_mask

def make_inf_matrix(inf_fun, inf_sampling, dm_mask, overpad=None):
    Nact = dm_mask.shape[0]
    Nacts = dm_mask.sum()
    Nsurf = int(inf_sampling * Nact)
    if overpad is None:
        overpad = round(inf_sampling)
    
    Nsurf_padded = Nsurf + overpad
    
    padded_inf_fun = pad_or_crop(inf_fun, Nsurf_padded)

    inf_cube = jnp.zeros((Nacts, Nsurf_padded, Nsurf_padded))
    act_inds = jnp.argwhere(dm_mask)

    for i in range(Nacts):
        Nx = int(jnp.round((act_inds[i][1] + 1/2 - Nact/2) * inf_sampling))
        Ny = int(jnp.round((act_inds[i][0] + 1/2 - Nact/2) * inf_sampling))
        shifted_inf_fun = shift(padded_inf_fun, (Ny,Nx))
        inf_cube = inf_cube.at[i].set(shifted_inf_fun)

    inf_matrix = inf_cube.reshape(Nacts, inf_cube.shape[1]**2,).T
    return inf_matrix

def map_command_to_actuators(command, dm_mask):
    actuators = command[dm_mask]
    return actuators

def map_actuators_to_command(act_vector, dm_mask):
        Nact = dm_mask.shape[0]
        command = jnp.zeros((Nact, Nact))
        command = command.at[dm_mask].set(act_vector)
        return command

def get_surf(actuators, 
             inf_matrix,
             inf_pixelscale=None, pixelscale=None):

    N = int(np.sqrt(inf_matrix.shape[0]))
    surf = inf_matrix.dot(actuators).reshape(N, N)

    if pixelscale is None and inf_pixelscale is None:
        return surf
    else:
        surf = interp_arr(surf, inf_pixelscale.to_value(u.m/u.pix), pixelscale.to_value(u.m/u.pix), order=1)
        return surf

def get_opd(actuators, 
            inf_matrix,
            inf_pixelscale=None, pixelscale=None):
    opd = 2*get_surf(actuators, inf_matrix, inf_pixelscale, pixelscale)
    return opd

def get_phasor(actuators, 
               inf_matrix,
               inf_pixelscale=None, pixelscale=None,
               wavelength=650e-9*u.m):
    opd = 2*get_surf(actuators, inf_matrix, inf_pixelscale, pixelscale)
    phasor = jnp.exp(1j*2*jnp.pi/wavelength.to_value(u.m) * opd)
    return phasor



