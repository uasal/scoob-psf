import numpy as np
import cupy as cp
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import ray

from .math_module import xp,_scipy, ensure_np_array
from . import imshows
import scoobpsf
module_path = Path(os.path.dirname(os.path.abspath(scoobpsf.__file__)))

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image


'''
FIXME: This file will eventually contain a compact model of SCOOB similar to how FALCO uses a compact model to compute Jacobians
'''

