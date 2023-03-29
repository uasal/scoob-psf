from . import scoobm, scoobm2, scoobi

if scoobi.scoobpy_avail:
    print('scoobpy installed: testbed interface available.')
else:
    print('scoobpy not installed: testbed interface unavailable.')

__version__ = '0.1.0'

