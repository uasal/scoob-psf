import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'

import matplotlib
from matplotlib.animation import FFMpegWriter, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['animation.html'] = 'html5'
plt.rcParams.update({'image.origin': 'lower',
                     'image.interpolation':'nearest'})

def create_anim(arrs):
    numframes = arrs.shape[0]
    print(numframes)

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5),dpi=125)
    
    im1 = ax.imshow(arrs[0,:,:],)
    # im1_title = ax.set_title(f'WFE: Time = {0.0:.2e}s', fontsize = 18)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im1, cax=cax)

    def animate(i):
        im1.set_data(arrs[i,:,:])
        # im1_title.set_text(f'WFE: Time = {wfe_times[i]:.2e}s')
        im1.set_clim(vmin=np.min(arrs[i,:,:]), vmax=np.max(arrs[i,:,:]))

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=numframes, )
    return anim


