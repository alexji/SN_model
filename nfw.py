import numpy as np
import cosmology
from scipy.integrate import quad

class NFW:
    def __init__(self,Mvir,zvir,cosm):
        # TODO find zvir, conc from Mvir
        self.kB = 1.38e-16 # erg/K
        self.Gc = 6.67e-8  # (cm/s)^2 cm/g
        self.mp = 1.67e-24 # g
        self.pc = 3.086e18 # cm
        self.kpc= 3.086e21 # cm
        self.yr = 3.156e7  # s
        self.Msun = 1.989e33 # g

        self.Mvir = Mvir # 1/h  Msun
        self.zvir = zvir
        self.cosm = cosm
        h = cosm.h
        self.mu = 0.59; mu = self.mu
        self.mbar = mu*self.mp

        self.conc = 4.8
        self.rho0 = 200./3 * self.conc**3/self.F(self.conc) * cosm.rho_c(zvir)

        ## These quantities are assuming conc = 4.8
        M8 = self.Mvir/(10**8)
        self.rvir = 0.76*self.kpc/h * M8**(1./3) * 10./(1+zvir) #cm
        self.vvir = 24e5 * M8**(1./3) * np.sqrt((1+zvir)/10.) #cm/s
        self.Tvir = 10**4.5 * M8**(2./3)*mu*(1+zvir)/10. #K
        self.vesc0 = 77.e5 * M8**(1./3) * np.sqrt((1+zvir)/10.) #cm/s
        self.rhoiso0 = cosm.rho_c(zvir) * 840./(h*h)

        self.rs = self.rvir/self.conc
        self.vmax = self.vvir * np.sqrt(0.2162*self.conc/(self.F(self.conc)))
        self.rvmax = 2.1626*self.rs

    def F(self,t):
        """ Proportional to integral of r^2 rho(r) """
        return np.log(1+t) - t/(1+t)

    def rho(self,r):
        x = r/self.rs
        return self.rho0/(x*(1+x)**2)
    def vc(self,r):
        return self.vvir * np.sqrt(self.F(r/self.rs)/(r/self.rvir * self.F(self.conc)))
    def vesc(self,r):
        x = r/self.rs
        return np.sqrt(2)*self.vvir*np.sqrt((self.F(x)+x/(1+x))/(r/self.rvir*self.F(self.conc)))
    def Mltr(self,r):
        return 4*np.pi*self.rho0*self.rs**3 * self.F(r/self.rs)

    def rho_isothermal(self,r):
        return self.rhoiso0*np.exp(-self.mbar/(2*self.kB*self.Tvir)*(self.vesc0**2 - self.vesc(r)**2))

if __name__=="__main__":
     c = cosmology.cosmology(Om=.3,Ol=.7,h=.7)
     M = 10**8
     zvir = 9
     nfw = NFW(M,zvir,c)
