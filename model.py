import numpy as np
from scipy import integrate
from scipy import interpolate
import cosmology
import nfw as NFW
from astropy.io import ascii


h=.5
Ob = .019/(h*h)
Om = .3
Ol = .7
fb = Ob/Om
zvir = 9
cosm = cosmology.cosmology(Om=Om,Ol=Ol,h=h)
Mvir = 10**8/cosm.h

rhoc = cosm.rho_c(zvir)
rhoIGM = fb*cosm.rho_m(zvir)
nfw = NFW.NFW(Mvir*cosm.h,zvir,cosm)

fstar = .01
Mstar = Mvir*fb*cosm.Om*fstar
imfMmax = 120
imfMSNe = 8
imfMmin = 0.1
def intsalpeter(Mmin,Mmax):
    return 0.74  * (Mmin**-1.35 - Mmax**-1.35)
def intsalpetermass(Mmin,Mmax):
    return 2.857 * (Mmin**-0.35 - Mmax**-0.35)
nSN = Mstar * intsalpeter(imfMSNe,imfMmax)/intsalpetermass(imfMmin,imfMmax)
ESN = 1e51 #erg
def ufunclike(f,x):
    return np.array(map(f, np.array(np.ravel(x))))

Mcrit = 33*8**(1.5)/(3.4*60**.5)
def tOB(M):
    def _tOB(M):
        if M <= Mcrit: return 33*(M/8)**-1.5 #* 1e6 * nfw.yr
        return 3.4 / np.sqrt(M/60) #* 1e6 * nfw.yr
    return ufunclike(_tOB,M)
tcrit = tOB(Mcrit)
def MOB(t): # t in Myr
    def _MOB(t):
        if t >= tcrit: return 8*33**(-2./3) * t**(-2./3)
        return 60*3.4**(-2) * t**(-2.)
    return ufunclike(_MOB,t)
def dMOBdt(t): # t in Myr
    def _dMOBdt(t):
        if t >= tcrit: return 8*33**(-2./3) * (2./3) * t**(-5./3)
        return 60*3.4**(-2) * (2) * t**(-3.)
    return ufunclike(_dMOBdt,t)
# Make L(t) smooth instead of stochastic
imfnorm = Mstar/integrate.quad(lambda x:x**-1.35,imfMmin,imfMmax)[0] # 1/Msun
def L(t):
    return ESN * imfnorm * (MOB(t))**(-2.35) * dMOBdt(t) / (1e6 * nfw.yr)

eta=6e-7 #cgs
C1 = 16*np.pi*nfw.mu*eta/(25*nfw.kB)
C2 = 125*np.pi*nfw.mbar/39

T0 = 10**4 #K
rho_rvir = nfw.rho_isothermal(nfw.rvir)
assert rho_rvir > rhoIGM
def rho0(r): #g/cc
    if r < nfw.rvir: return nfw.rho_isothermal(r)
    return (rho_rvir-rhoIGM)*np.exp(-(r-nfw.rvir)/(1.2*nfw.rvir))+rhoIGM
def rho0p(r): #derivative of rho0
    raise NotImplementedError
def P0(r): #cgs: erg/cm^3
    return rho0(r)*nfw.kB*T0/nfw.mbar

def Pb(E,R):
    return E/(2*np.pi*R**3)
def nbar(R,E,T):
    return 5./3 * Pb(E,R)/(kB * T) #5/3 n_c

def vdot(y,t): #Rdotdot
    v,R,E,T = y
    return 3*(Pb(E,R)-P0(R))/(R*rho0(R)) - 3*Gc*nfw.Mltr(R)*rho0(R)*v/(4*np.pi*R**5) - 4*v*v/R - v*v*rho0p(R)/rho0(R)

def Rdot(y,t):
    return y[0]

def Edot(y,t):
    v,R,E,T = y
    return L(t)-4*np.pi*R*R*Pb(E,R)*v-4*np.pi/3 * nbar(R,E,T) * Lambda(T)

def Tdot(y,t):
    v,R,E,T = y
    return 3*T*v/R + T*Pdot(y,t)/Pb(E,R) - (C1/C2)*T**(9/2.)*nfw.kB/(R*R*Pb(E,R))
def Pdot(y,t):
    v,R,E,T = y
    return Edot(y,t)/(2*np.pi*R**3) - 3*E*v/(2*np.pi*R**4)

coolingtable = 'cooling/m-30.cie'
tab = ascii.read(coolingtable,header_start=1)
_Lambda = interpolate.interp1d(np.concatenate(([3.93],tab['log(T)'])),np.concatenate(([-28],tab['log(lambda norm)'])))
def Lambda(T):
    # Tbar = 5/7 T_c
    return 10**_Lambda(np.log10(T)-.146128)


if __name__=="__main__":
    pass
