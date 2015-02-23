import numpy as np
from scipy import integrate
from scipy import interpolate
import cosmology
import nfw as NFW
from astropy.io import ascii

import sedov

import pylab as plt
#from pylab import *
#import seaborn as sns

def ufunclike(f,x):
    return np.array(map(f, np.array(np.ravel(x))))

coolingtable = 'cooling/m-30.cie'
tab = ascii.read(coolingtable,header_start=1)
__Cooling = interpolate.interp1d(np.concatenate(([3.93],tab['log(T)'])),np.concatenate(([-28],tab['log(lambda norm)'])))
def _Cooling(logT):
    if logT<=(3.93+.146128): return 1e-28
    return __Cooling(logT-.146128)
def Cooling(T):
    # Tbar = 5/7 T_c
    return 10**ufunclike(_Cooling,np.log10(T))

#def _L(t,ESN=1.e51):
#    """ t in Myr """
#    if t < 0.1: return ESN/(1.e5 * 3.e7) #erg/s
#    return 0
#def L(t):
#    return ufunclike(_L,t)
def _zero(t):
    return 0
def L(t):
    return ufunclike(_zero,t)

h = 0.7
Om = .3; Ol = 0.7; Ob = .05; fb = Ob/Om
Mvir = 1e6 #Msun
zvir = 25
c = 3.0
cosm = cosmology.cosmology(Om=Om,Ol=Ol,h=h)
nfw = NFW.NFW(Mvir,zvir,c,cosm)

Rvir = nfw.Rvir * nfw.kpc

T0 = 10**4 #K
rho_rvir = nfw.rho_isothermal(nfw.Rvir)
rhoIGM = cosm.rho_m(zvir)
assert rho_rvir > rhoIGM
#def _rho0(r): #g/cc
#    if r < Rvir: return nfw.rho_isothermal(r/nfw.kpc)
#    return (rho_rvir-rhoIGM)*np.exp(-(r-Rvir)/(1.2*Rvir))+rhoIGM
rho_flat = 10**(-1.5)*nfw.mbar
def _rho0(r):
    if r < Rvir: return rho_flat
    return (rho_flat-rhoIGM)*np.exp(-(r-Rvir)/(1.2*Rvir))+rhoIGM
def rho0(r):
    return ufunclike(_rho0,r)
_rarr = np.arange(0,300*nfw.Rvir,.001)[1:]*nfw.kpc
_rhoarr = rho0(_rarr)
_rho0interp = interpolate.UnivariateSpline(_rarr, _rhoarr, s=0)
_rho0pinterp= _rho0interp.derivative()

def _rho0p(r):
    out = np.abs(_rho0pinterp(r))
    if out < 1e-35: return 0
    return out
def rho0p(r): #derivative of rho0
    return ufunclike(_rho0p,r)
def P0(r): #cgs: erg/cm^3
    return rho0(r)*nfw.kB*T0/nfw.mbar

def Pb(E,R):
    return E/(2*np.pi*R**3)
def nbar(R,E,T):
    return 5./3 * Pb(E,R)/(nfw.kB * T) #5/3 n_c

def vdot(y,t): #Rdotdot
    v,R,E,T = y
    return 3*(Pb(E,R)-P0(R))/(R*rho0(R)) - 3*nfw.Gc*nfw.Msun*nfw.Mltr(R)*rho0(R)*v/(4*np.pi*R**5) - 4*v*v/R - v*v*rho0p(R)/rho0(R)

def Rdot(y,t):
    return y[0]

def Edot(y,t):
    v,R,E,T = y
    return L(t)-4*np.pi*R*R*Pb(E,R)*v-4*np.pi/3 * nbar(R,E,T) * Cooling(T)

eta=6e-7 #cgs
C1 = 16*np.pi*nfw.mbar*eta/(25.*nfw.kB)
C2 = 125*np.pi*nfw.mbar/39.
def Tdot(y,t):
    v,R,E,T = y
    return 3*T*v/R + T*Pdot(y,t)/Pb(E,R) - (2.3)*(C1/C2)*T**(4.5)*nfw.kB/(R*R*Pb(E,R))
def Pdot(y,t):
    v,R,E,T = y
    return Edot(y,t)/(2*np.pi*R**3) - 3*E*v/(2*np.pi*R**4)

def ydot(y,t):
    return [vdot(y,t),Rdot(y,t),Edot(y,t),Tdot(y,t)]

def run_model(Ei,tmax,dt=.005):
    st = sedov.sedov(nfw.rhoiso0,Ei)
    ti = 1000.*3e7 #10,000 years
    Ri = st.Rs(ti); vi = st.us(ti)
    Ti = Pb(Ei,Ri)/(nfw.kB * (nfw.rhoiso0/nfw.mbar))
    if Ti > 10**8.5: 
        print "WARNING: Ti > 1e8.5, setting to 1e8.5"
        Ti = 10**8.5

    dt = .005
    tmin = 0
    tarr = np.arange(tmin,tmax+dt,dt)*(3e7 * 1e6)
    # setup initial conditions
    y0 = [vi,Ri,Ei,Ti] #cgs units
    print y0
    print ydot(y0,0)

    # integration
    yout,out = integrate.odeint(ydot, y0, tarr,full_output=True)
    #args=(), Dfun=None, col_deriv=0, full_output=0, ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0

    tarr = tarr/(3e7 * 1e6)
    varr = yout[:,0]/1e5 
    Rarr = yout[:,1]/nfw.kpc
    Earr = yout[:,2]/1e51
    Tarr = yout[:,3]
    ii = varr >= 0
    tarr = tarr[ii]; varr = varr[ii]; Rarr = Rarr[ii]; Earr = Earr[ii]; Tarr = Tarr[ii]
    return tarr,varr,Rarr,Earr,Tarr


def plot_model(tarr,varr,Rarr,Earr,Tarr,fig=None):
    if fig==None:
        fig,axarr = plt.subplots(2,2,figsize=(8,8))
    else:
        axarr = np.reshape(fig.axes,(2,2))

    axarr[0,0].plot(tarr,Rarr)
    axarr[0,0].set_xlabel('t (Myr)')
    axarr[0,0].set_ylabel('R (kpc)')
    axarr[0,1].plot(tarr,varr)
    axarr[0,1].set_xlabel('t (Myr)')
    axarr[0,1].set_ylabel('v (km/s)')
    axarr[0,1].set_yscale('log')
    axarr[1,0].plot(tarr,Earr)
    axarr[1,0].set_xlabel('t (Myr)')
    axarr[1,0].set_ylabel('E_51')
    axarr[1,0].set_yscale('log')
    axarr[1,1].plot(tarr,Tarr)
    axarr[1,1].set_xlabel('t (Myr)')
    axarr[1,1].set_ylabel('T [k]')
    axarr[1,1].set_yscale('log')

if __name__=="__main__":
    fig,axes = plt.subplots(2,2,figsize=(10,10))
    fig.subplots_adjust(wspace=.25)
    tarr,varr,Rarr,Earr,Tarr = run_model(1e51,6)
    plot_model(tarr,varr,Rarr,Earr,Tarr,fig=fig)
    tarr,varr,Rarr,Earr,Tarr = run_model(3e51,6)
    plot_model(tarr,varr,Rarr,Earr,Tarr,fig=fig)
    tarr,varr,Rarr,Earr,Tarr = run_model(1e52,12)
    plot_model(tarr,varr,Rarr,Earr,Tarr,fig=fig)
    ax = axes[1,1]; ax.legend(['1e51','3e51','1e52'],loc='upper right')
    plt.show()
