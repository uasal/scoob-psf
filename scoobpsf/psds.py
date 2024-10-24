from .math_module import xp, _scipy, ensure_np_array

import numpy as np
import scipy
import astropy.units as u
import copy

def generate_freqs(Nf=2**18+1, 
                   f_min=0*u.Hz, 
                   f_max=100*u.Hz,
                   ):
    """_summary_

    Parameters
    ----------
    Nf : Number of samples for the frequency range, optional
         must be supplied as a power of 2 plus 1, by default 2**18+1
    f_min : _type_, optional
        _description_, by default 0*u.Hz
    f_max : _type_, optional
        _description_, by default 100*u.Hz

    Returns
    -------
    _type_
        _description_
    """
    if bin(Nf-1).count('1')!=1: 
        raise ValueError('Must supply number of samples to be a power of 2 plus 1. ')
    del_f = (f_max - f_min)/Nf
    freqs = np.arange(f_min.value, f_max.value, del_f.value) * u.Hz
    Nt = 2*(Nf-1)
    del_t = (1/(2*f_max)).to(u.s)
    times = np.linspace(0, (Nt-1)*del_t, Nt)
    return freqs, del_f, times


def kneePSD(freqs,beta,fn,alpha):
    psd = beta/(1+freqs/fn)**alpha
    try:
        psd.decompose()
        return psd
    except:
        return psd


def generate_time_series(psd, f_max, rms=None,  seed=123,):
    Nf = len(psd)
    Nt = 2*(Nf-1)
    del_t = (1/(2*f_max)).to(u.s)
    times = np.linspace(0, (Nt-1)*del_t, Nt)

    P_fft_one_sided = copy.copy(psd)

    N_P = len(P_fft_one_sided)  # Length of PSD
    N = 2*(N_P - 1)
    # print(N_P, N)

    # Because P includes both DC and Nyquist (N/2+1), P_fft must have 2*(N_P-1) elements
    P_fft_one_sided[0] = P_fft_one_sided[0] * 2
    P_fft_one_sided[-1] = P_fft_one_sided[-1] * 2
    P_fft_new = np.zeros((N,), dtype=complex)
    P_fft_new[0:int(N/2)+1] = P_fft_one_sided
    P_fft_new[int(N/2)+1:] = P_fft_one_sided[-2:0:-1]

    X_new = np.sqrt(P_fft_new)

    # Create random phases for all FFT terms other than DC and Nyquist
    np.random.seed(seed)
    phases = np.random.uniform(0, 2*np.pi, (int(N/2),))

    # Ensure X_new has complex conjugate symmetry
    X_new[1:int(N/2)+1] = X_new[1:int(N/2)+1] * np.exp(2j*phases)
    X_new[int(N/2):] = X_new[int(N/2):] * np.exp(-2j*phases[::-1])
    X_new = X_new * np.sqrt(N) / np.sqrt(2)

    # This is the new time series with a given PSD
    x_new = scipy.fft.ifft(X_new)

    if rms is not None: 
        x_new *= rms/np.sqrt(np.mean(np.square(x_new)))
    # print(np.sqrt(np.sum(np.square(x_new.real))), np.sqrt(np.sum(np.square(x_new.imag))))

    return x_new.real

import matplotlib.pyplot as plt

def plot(freqs, psd, plot_integral=False):
    plt.plot(freqs,psd)
    plt.title(f"temporal PSD")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.xlabel(freqs.unit)
    plt.show()

    if plot_integral:
        Nints = 2000
        Nf = len(freqs)
        del_f = freqs[1]-freqs[0]
        int_freqs = []
        psd_int = []
        for i in range(Nints):
            i_psd = int(np.round(Nf/(Nints-i)))
            int_freqs.append(freqs[i_psd-1].to_value(u.Hz))
            psd_int.append(np.trapz(psd[:i_psd])*del_f.to_value(u.Hz))

        plt.plot(int_freqs, psd_int)
        plt.title(f"Integral of temporal PSD over frequency range")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid()
        plt.xlabel(freqs.unit)
        plt.show()

