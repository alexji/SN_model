import numpy as np
from scipy.integrate import odeint
from nfw import NFW
import cosmology
from scipy.interpolate import interp1d

from astropy.io import ascii

## Everything done in cgs units
Gc = 6.67e-8
kB = 1.38e-16
kpc = 3.086e21
Msun= 2.e33
yr = 3.16e7
mp = 1.67e-24

cosm = cosmology.cosmology(h=.7,Om=.3,Ol=.7)
nfw = NFW(1.e6,25,None,cosm)
#nfw = NFW(5.e5,20,None,cosm)

tau_star = 3e6 * yr
Tion = 3.e4
Tvir = nfw.Tvir
cion = np.sqrt(kB*Tion/(0.59 * mp))
cvir = np.sqrt(kB*Tvir/(1.23 * mp))

Shu = ascii.read('Shu_data.tab')
alpha_eps = interp1d(Shu['eps'],Shu['alpha0'],kind='quadratic')
xsh_eps = interp1d(Shu['eps'],Shu['xs'],kind='quadratic')

def ydot(y,x):
    a,v = y
    denom = (v-x)**2 - 1.
    return [a*(a-2./x * (x-v))*(x-v)/denom, ((x-v)*a-2./x)*(x-v)/denom]

def rho_(alpha,t):
    return alpha/(4*np.pi*Gc*t*t)
def n_(alpha,t,X=.75):
    return X*rho_(alpha,t)/mp

def solve(eps,xmin=.01,xmax=None):
    _a0 = alpha_eps(eps)
    if xmax==None: xmax = xsh_eps(eps)
    a0 = _a0 + (_a0/6.) * (2./3 - _a0) * xmin**2
    v0 = 2./3 * xmin + 1./45 * (2./3 - _a0)*xmin**3
    y0 = [a0,v0]

    xarr = np.linspace(xmin,xmax,1000)
    yarr = odeint(ydot,y0,xarr)
    return xarr,yarr

def get_eps(Tvir,Tion):
    return (.59/1.23) * (Tvir/Tion)

if __name__=="__main__":
    import pylab as plt
    plt.figure()
    eps = get_eps(Tvir,Tion)
    x,y = solve(eps)
    savedat = np.zeros((len(x),3))
    savedat[:,0] = x; savedat[:,1] = y[:,0]; savedat[:,2] = y[:,1]
    np.save('champagne.npy',savedat)
    for tau_star in [1.e6*yr,3.e6*yr,1.e7*yr]:
        r = cion*x*tau_star
        n = n_(y[:,0],tau_star)
        plt.plot(r/(kpc/1000.),n)
    plt.loglog()
    plt.show()
