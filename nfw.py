import numpy as np
import cosmology
from scipy.integrate import quad
import commah # Correa+2015b

class NFW:
    def __init__(self,Mvir,zvir,c,cosm):
        self.Mvir = Mvir # Msun
        self.zvir = zvir
        self.c = c
        self.cosm = cosm
        fb = 1./6
        
        if c==None:
            print "Automatically generating concentration from COMMAH"
            output = commah.run("Planck13",zi=zvir,Mi=Mvir,z=[zvir]) # TODO use cosm
            c = np.ravel(output['c'])[0]
            self.c = c

        self.Msun = 2.e33   # g
        self.yr = 3.156e7  # s
        self.pc = 3.086e18 # cm
        self.kpc = 3.086e21 # cm
        self.kB = 1.38e-16 # erg/K
        self.Gc = 6.67e-8  # (cm/s)^2 cm/g
        self.mu = 0.59 # for ionized primordial gas
        self.mp = 1.67e-24 # g

        self.Delta_c = 200.
        self.rhoc = cosm.rho_c(zvir) # g/cc
        self.rhovir = self.Delta_c*self.rhoc # g/cc
        self.Rvir = (3*Mvir*self.Msun/(4*np.pi*self.rhovir))**(1./3) / self.kpc # kpc
        self.rs = self.Rvir/c # kpc
        self.rho0 = self.rhovir*c*c*c/(self.F(c)*3.) # g/cc

        self.Vvir = np.sqrt(self.Gc * Mvir * self.Msun / (self.Rvir*self.kpc)) / 1.e5 #km/s
        self.Vmax = self.Vvir * np.sqrt(0.2162*c/(self.F(c))) # km/s
        self.RVmax = 2.1626*self.rs # kpc TODO check
        self.Tvir = self.mp*(self.Vvir*1e5)**2/(2. * self.kB) * self.mu

        # for isothermal density profile
        self.Vesc0 = np.sqrt(2)*self.Vvir*np.sqrt(self.c/self.F(self.c))
        self.mbar = self.mu*self.mp
        A = 2*c/self.F(c)
        denom = quad(lambda t:(1+t)**(A/t)*t*t,0,c)[0]
        self.rhoiso0 = cosm.rho_c(zvir) * (200./3)*(c**3)*fb*cosm.Om*np.exp(A)/denom

    def F(self,t):
        """ proportional to integral of r^2 rho(r) """
        return np.log(1.+t)-t/(1.+t)

    def rho(self,r):
        """ r in kpc, returns g/cc """
        x = r/self.rs
        return self.rho0/(x*(1.+x)**2)

    def Mltr(self,r):
        """ r in kpc, returns Msun """
        return 4*np.pi*self.rho0*(self.rs*self.kpc)**3 * self.F(r/self.rs) / self.Msun

    def Vc(self,r):
        return self.Vvir * np.sqrt(self.F(r/self.rs)/(r/self.Rvir * self.F(self.c)))

    def Vesc(self,r):
        x = r/self.rs
        return np.sqrt(2)*self.Vvir*np.sqrt((self.F(x)+x/(1+x))/(r/self.Rvir*self.F(self.c)))

    def rho_isothermal(self,r):
        return np.exp(np.log(self.rhoiso0) - self.mbar*(1.e10)*(self.Vesc0**2 - self.Vesc(r)**2)/(2*self.kB*self.Tvir))

if __name__=="__main__":
     cosm = cosmology.cosmology(Om=.3,Ol=.7,h=.7)
     M = 10**8
     zvir = 9
     c = 4.8
     nfw = NFW(M,zvir,c,cosm)
