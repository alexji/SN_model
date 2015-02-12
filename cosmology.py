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
    def invE(self,z): #1/H(z), but unitless
        return 1.0/np.sqrt(self.Om*(1+z)**3 + self.Ol + self.Ok*(1+z)**2 + self.Or*(1+z)**4)
    def H(self,z): #km/s/Mpc
        return self.h*100/self.invE(z)
    def t_lookback(self,z): #in Gyr
        return 9.785/self.h*quad(lambda x: self.invE(x)/(1+x),0,z)[0]
    def t_age(self,z): #in Gyr
        return self.age - self.t_lookback(z)

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

if __name__=="__main__":
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
