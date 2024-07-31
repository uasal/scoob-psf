import numpy as np
import scipy
from scipy import integrate

import astropy.units as u
from astropy.constants import h, c, k_B, R_sun
from astropy.io import fits

import matplotlib.pyplot as plt

class SOURCE():
    
    def __init__(self,
                 wavelengths,
                 distance,
                 temp,
                 diameter,
                 lambdas=None,
                 name='UNNAMED',
                ):
        self.name = name
        self.wavelengths = wavelengths.to(u.m)
        self.distance = distance.to(u.m)
        self.temp = temp.to(u.K)
        self.diameter = diameter.to(u.m)
        
        dist = self.distance.to_value(u.m).astype(np.float128)
        R = self.diameter.to_value(u.m).astype(np.float128)

        self.solid_angle = 2*np.pi * (1 - np.sqrt(dist**2 - R**2)/dist) * u.sr # look up solid angle for a star
        print(self.solid_angle)
        
        self.lambdas = np.linspace(400, 1000, 12001)*1e-9*u.m if lambdas is None else lambdas
        self.del_lam = self.lambdas[1] - self.lambdas[0]
        self.energies = h*c/self.lambdas / u.photon
        
        self.spectrum = 2*h*c**2/(self.lambdas**5)/(np.exp(h*c/(k_B*self.temp*self.lambdas)) - 1)/u.sr
        self.spectrum = self.spectrum.to(u.J/u.s/u.sr/u.m**2/u.nm) * self.solid_angle
        
        self.spectrum_ph = self.spectrum / self.energies
        self.spectrum_ph = self.spectrum_ph.to(u.ph/u.s/u.m**2/u.nm)
        
#         self.spectrum_ph = self.spectrum / self.energies * self.del_lam
#         self.spectrum_ph = self.spectrum_ph.to(u.ph/u.s/u.m**2)
        
    def plot_spectrum(self):
        lam_min = np.min(self.lambdas.to_value(u.nm))
        lam_max = np.max(self.lambdas.to_value(u.nm))
        spec_min = np.min(self.spectrum.to_value(u.W/u.m**2/u.nm))
        spec_max = np.max(self.spectrum.to_value(u.W/u.m**2/u.nm))

        wavelengths = self.wavelengths.to_value(u.nm)
        del_waves = wavelengths[1]-wavelengths[0]

        fig,ax = plt.subplots(1,1,dpi=125)
        ax.plot(self.lambdas.to(u.nm), self.spectrum.to(u.W/u.m**2/u.nm))
        ax.set_xlim([lam_min, lam_max])
        ax.set_ylim([spec_min-0.1*spec_min, spec_max+0.1*spec_min])
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel('$W/m^2/nm$')
        ax.set_title(f'Flux for {self.name}')
        
        wavelength_c = (wavelengths[-1] + wavelengths[0])/2
        for i in range(len(wavelengths)):
            r = 0.6 - 3*(1-wavelengths[i]/wavelength_c)
            b = 0.6 + 3*(1-wavelengths[i]/wavelength_c)
            if wavelengths[i]<wavelength_c:
                g = 0.8 - 5*(1-wavelengths[i]/wavelength_c)
            else:
                g = 0.8 + 5*(1-wavelengths[i]/wavelength_c)
            alpha = 0.8
            color = (r,g,b,alpha)
            plt.axvspan(wavelengths[i]-del_waves/2, wavelengths[i]+del_waves/2, color=color)
            
    def plot_spectrum_ph(self, save_fig=None):
        lam_min = np.min(self.lambdas.to_value(u.nm))
        lam_max = np.max(self.lambdas.to_value(u.nm))
        spec_min = np.min(self.spectrum_ph.to_value(u.ph/u.s/u.m**2/u.nm))
        spec_max = np.max(self.spectrum_ph.to_value(u.ph/u.s/u.m**2/u.nm))
        
        wavelengths = self.wavelengths.to_value(u.nm)
        del_waves = wavelengths[1]-wavelengths[0]
        
        fig,ax = plt.subplots(1,1,dpi=125)
        ax.plot(self.lambdas.to(u.nm), self.spectrum_ph.to(u.ph/u.s/u.m**2/u.nm))
        ax.set_xlim([lam_min, lam_max])
        ax.set_ylim([spec_min-0.1*spec_min, spec_max+0.1*spec_min])
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel('$ph/s/m^2/nm$')
        ax.set_title(f'Photon Flux for {self.name}')
        
        wavelength_c = (wavelengths[-1] + wavelengths[0])/2
        for i in range(len(wavelengths)):
            r = 0.6 - 3*(1-wavelengths[i]/wavelength_c)
            b = 0.6 + 3*(1-wavelengths[i]/wavelength_c)
            if wavelengths[i]<wavelength_c:
                g = 0.8 - 5*(1-wavelengths[i]/wavelength_c)
            else:
                g = 0.8 + 5*(1-wavelengths[i]/wavelength_c)
            alpha = 0.8
            color = (r,g,b,alpha)
            plt.axvspan(wavelengths[i]-del_waves/2, wavelengths[i]+del_waves/2, color=color)

        if save_fig is not None:
            fig.savefig(save_fig, format='pdf', bbox_inches="tight")
        
    def calc_fluxes(self):
        Nwaves = len(self.wavelengths)
        wavelengths = self.wavelengths.to(u.nm)
        del_waves = wavelengths[1]-wavelengths[0]
        
        lambdas = self.lambdas.to(u.nm)
        
        spectrum = self.spectrum.to(u.J/u.s/u.m**2/u.nm)
        fluxes = []
        for i in range(Nwaves):
            wavelength = wavelengths[i]
            min_wave = wavelength-del_waves/2
            max_wave = wavelength+del_waves/2
            
            inds = np.where((lambdas>min_wave)&(lambdas<max_wave))

            lambdas_domain = lambdas[inds].value
            spec_range = spectrum[inds].value

            flux = integrate.simpson(spec_range, lambdas_domain)*u.J/u.s/u.m**2 # integrate over the wave band
            E = h*c/wavelength / u.photon # energy of the photon for this wavelength [J/photon]
            
            flux_in_ph = (flux/E).to_value(u.ph/u.s/u.m**2) # convert to photons/s/m**2
            fluxes.append(flux_in_ph)
        
        fluxes = np.array(fluxes)*u.ph/u.s/u.m**2
        return fluxes
    
    def calc_fluxes_ph(self): # use each wavelengths photon flux for the calculation
        Nwaves = len(self.wavelengths)
        wavelengths = self.wavelengths.to(u.nm)
        del_waves = wavelengths[1]-wavelengths[0]
        
        lambdas = self.lambdas.to(u.nm)
        
        spectrum_ph = self.spectrum_ph.to(u.ph/u.s/u.m**2/u.nm)
#         spectrum_ph = (self.spectrum_ph/self.del_lam).to(u.ph/u.s/u.m**2/u.nm)
        fluxes = []
        for i in range(Nwaves):
            wavelength = wavelengths[i]
            min_wave = wavelength-del_waves/2
            max_wave = wavelength+del_waves/2
            
            inds = np.where((lambdas>min_wave)&(lambdas<max_wave))

            lambdas_domain = lambdas[inds].value
            spec_range = spectrum_ph[inds].value

            flux = integrate.simpson(spec_range, lambdas_domain)*u.ph/u.s/u.m**2 # integrate over the wave band
            fluxes.append(flux.value)
        
        fluxes = np.array(fluxes)*u.ph/u.s/u.m**2
        return fluxes


