import numpy as np
import scipy
import poppy

class np_backend:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)
    
class scipy_backend:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)

if poppy.accel_math._USE_CUPY:
    import cupy as cp
    import cupyx.scipy
    xp = np_backend(cp)
    _scipy = scipy_backend(cupyx.scipy)
else:
    xp = np_backend(np)
    _scipy = scipy_backend(scipy)

def update_xp(module):
    xp._srcmodule = module
    
def update_scipy(module):
    _scipy._srcmodule = module
    
def ensure_np_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return arr.get()
        
        
        
        