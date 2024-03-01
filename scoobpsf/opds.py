import numpy as np
import poppy
import astropy.units as u

SURF_UNIT = u.nm
LENGTH_UNIT = u.m
SCREEN_SIZE = int(800*4)
SCREEN_SIZE = 5080
SCREEN_SIZE = 8192

SEEDS = np.arange(1234, 1234+100)

oap_params = {
    'psd_parameters': [[3.029, 329.3* u.nm**2/(u.m**(3.029-2)), 0.019 * u.m, 0.0001, 0 * u.nm**2 * u.m**2],
                    [-3.103, 1.606e-12 * u.nm**2/(u.m**(-3.103-2)), 16 * u.m, 0.00429,0 * u.nm**2 * u.m**2],
                    [0.8, 0.0001 * u.nm**2/(u.m**(0.8-2)), 0.024 * u.m, 0.00021, 6.01284e-13 * u.nm**2 * u.m**2]], 
    'psd_weight': [1],
    'apply_reflection':True, 
    'screen_size':SCREEN_SIZE, 
}

flat_params = {
    'psd_parameters': [[3.284, 1180 * u.nm**2/(u.m**(3.284-2)), 0.017 * u.m, 0.0225, 0 * u.nm**2 * u.m**2],
                        [1.947, 0.02983 * u.nm**2/(u.m**(1.947-2)), 15 * u.m, 0.00335, 0 * u.nm**2 * u.m**2],
                        [2.827, 44.25 * u.nm**2/(u.m**(2.827-2)), 0.00057 * u.m, 0.000208, 1.27214e-14 * u.nm**2 * u.m**2]], 
    'psd_weight': [1],
    'apply_reflection':True, 
    'screen_size':SCREEN_SIZE, 
}

incident_angles = {
    'oap1': 0.0*u.degree,
}

wfe_psds = {
    'oap1':poppy.PowerSpectrumWFE(**oap_params, incident_angle=incident_angles['oap1'], seed=SEEDS[0]), 
}


