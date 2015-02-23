import numpy as np
from scipy.integrate import quad

class cosmology:
    def __init__(self,Om=.3183,Ol=.6817,Or=9.35*10**-5,h=.67,put_in_h=True):
        # default is Planck cosmological parameters
        # Or = Om/(1+zeq), zeq = 3403
        self.put_in_h = put_in_h
        if put_in_h:
            self.h = h
        else:
            self.h = 1.0
        self.Om = Om
        self.Ol = Ol
        self.Or = Or
        self.Ok = 1 - Om - Ol - Or
        self.flat = np.abs(self.Ok) < .001 # this is close enough to flat
        self.age = self.t_lookback(10000)
        if self.flat:
            #self.a0 = 1.0 # doesn't matter though..
            self.Sk = lambda x: x
            self.rootOk = 1.0
        else:
            #self.a0 = 3000/(self.h * np.sqrt(np.abs(self.Ok))) #np.abs(Ok) = (c/Ha)^2
            self.rootOk = np.sqrt(np.abs(self.Ok))
            if self.Ok > 0:
                self.Sk = lambda x: np.sinh(x)
            else:
                self.Sk = lambda x: np.sin(x)

    # times: H, t_lookback, t_age
    def E(self,z):
        return np.sqrt(self.Om*(1+z)**3 + self.Ol + self.Ok*(1+z)**2 + self.Or*(1+z)**4)
    def invE(self,z): #1/H(z), but unitless
        return 1.0/np.sqrt(self.Om*(1+z)**3 + self.Ol + self.Ok*(1+z)**2 + self.Or*(1+z)**4)
    def H(self,z): #km/s/Mpc
        return self.h*100*self.E(z)
    def t_lookback(self,z): #in Gyr
        return 9.785/self.h*quad(lambda x: self.invE(x)/(1+x),0,z)[0]
    def t_age(self,z): #in Gyr
        return self.age - self.t_lookback(z)

    def BN98(self,z):
        """ Bryan and Norman 1998 virial overdensity, assuming Omega_k = 0 """
        assert(self.flat)
        x = self.Om * (1.+z)**3. * self.invE(z)**2. - 1.
        return 18*np.pi*np.pi + 82*x - 39*x*x

    # distances: d_comoving, d_luminosity, d_angular
    def integrate_invE(self,z):
        return quad(lambda x: self.invE(x),0,z)[0]
    def d_comoving(self,z):
        return 3000./(self.h*self.rootOk) * self.Sk(self.rootOk*self.integrate_invE(z))
    def d_angular(self,z):
        return self.d_comoving(z)/(1+z)
    def d_luminosity(self,z):
        return self.d_comoving(z)*(1+z)

    # densities, g/cc
    def rho_c(self,z):
        return self.H(z)**2 * 1.878 * 10**(-33.)
    def rho_m(self,z):
        return self.Om*self.rho_c(z)
    def rho_l(self,z):
        return self.Ol*self.rho_c(z)

    # Prada+2012 concentration
    def Prada_conc(self,z,M):
        a = 1./(1+z)
        x = (self.Ol/self.Om)**(1./3)*a
        D = 2.5*(self.Om/self.Ol)**(1./3) * np.sqrt(1+x**3)/x**1.5 * quad(lambda xx: xx**1.5/(1+xx**3)**1.5,0,x)[0]
        
        y = (1.e12)/(M)
        sigma = D*16.9*(y**.41)/(1+1.102*(y**.2)+6.22*(y**.333))
        return self._Prada_conc(x,sigma)
    def _Prada_conc(self,x,sigma):
        c0 = 3.681; c1 = 5.033; alpha = 6.948; x0 = 0.424
        def cmin(xx):
            return c0 + (c1-c0)*(np.arctan(alpha*(xx-x0))/np.pi + .5)
        siginv0 = 1.047; siginv1 = 1.646; beta = 7.386; x1 = 0.526
        def siginvmin(xx):
            return siginv0 + (siginv1-siginv0)*(np.arctan(beta*(xx-x1))/np.pi + .5)
        B0 = cmin(x)/cmin(1.393)
        B1 = siginvmin(x)/siginvmin(1.393)

        sigp = B1*sigma
        A = 2.881; b = 1.257; c = 1.022; d = 0.060
        Csigp = A*((sigp/b)**c + 1)*np.exp(d/(sigp**2))
        return B0 * Csigp

if __name__=="__main__":
    import pylab as plt
    cosm = cosmology(h=.70,Ol=.73,Om=.27)
    Monh = np.logspace(10.5,15)
    M = Monh/cosm.h

    plt.figure()
    plt.plot(Monh,np.log10(map(lambda MM: cosm.Prada_conc(0,MM),  M)),'k')
    plt.plot(Monh,np.log10(map(lambda MM: cosm.Prada_conc(0.5,MM),M)),'orange')
    plt.plot(Monh,np.log10(map(lambda MM: cosm.Prada_conc(1,MM),  M)),'green')
    plt.plot(Monh,np.log10(map(lambda MM: cosm.Prada_conc(2,MM),  M)),'red')
    plt.plot(Monh,np.log10(map(lambda MM: cosm.Prada_conc(3,MM),  M)),'cyan')
    plt.gca().set_xscale('log')
    plt.ylim((.55,1))
    plt.xlabel('log M200 (h^-1 Msun)')
    plt.ylabel('log c200')
    plt.show()
def old():
    c1 = cosmology(Om=1,Ol=0,h=0.7,put_in_h=False)
    c2 = cosmology(Om=.25,Ol=0,h=0.7,put_in_h=False)
    c3 = cosmology(Om=0.27,Ol=0.73,h=0.7,put_in_h=False)
    c4 = cosmology(put_in_h=False)
    import pylab as plt
    plt.figure()
    zarr = np.linspace(0,10)

    for c in [c1,c2,c3]:
        dcom = [c.d_comoving(z)/3000. for z in zarr]
        dlum = [c.d_luminosity(z)/3000. for z in zarr]
        dang = [c.d_angular(z)/3000. for z in zarr]
        hubb = [c.H(z)/100 for z in zarr]

        plt.subplot(221)
        plt.plot(zarr,hubb); plt.ylabel('hubble')
        plt.subplot(222)
        plt.plot(zarr,dcom); plt.ylabel('d_com')
        plt.subplot(223)
        plt.plot(zarr,dlum); plt.ylabel('d_lum')
        plt.subplot(224)
        plt.plot(zarr,dang); plt.ylabel('d_ang')
    plt.show()
