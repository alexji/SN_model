import numpy as np
from scipy import integrate
from scipy import interpolate
import cosmology
import nfw as NFW
from astropy.io import ascii

import sedov

import pylab as plt
import cPickle as pickle
#from pylab import *
#import seaborn as sns

def ufunclike(f,x):
    return np.array(map(f, np.array(np.ravel(x))))
def _zero(t):
    return 0
def L(t):
    return ufunclike(_zero,t)

class MFR01(object):
    def __init__(self,Mvir,zvir,c=None,h=0.7,Om=0.3,Ol=0.7,Ob=0.05,
                 L=L,n_flat=0.1,useiso=True,useflat=False,usechampagne=False,T0 = 10**4):
        self.R_SN = {}
        self.t_SN = {}

        self.Mvir = Mvir; self.zvir = zvir; self.c = c
        self.h = h; self.Om = Om; self.Ol = Ol; self.Ob = Ob
        fb = Ob/Om; self.fb = fb
        self.L = L

        # Set up cooling table
        coolingtable = 'cooling/m-30.cie'
        tab = ascii.read(coolingtable,header_start=1)
        _CoolingInt = interpolate.interp1d(np.concatenate(([3.93],tab['log(T)'])),np.concatenate(([-28],tab['log(lambda norm)'])))
        def _Cooling(logT):
            if logT<=(3.93+.146128): return -28      # extrapolate at flat
            if logT>(8.5+.146128): return -22.47 # extrapolate at flat
            return _CoolingInt(logT-.146128)
        def Cooling(T):
            # Tbar = 5/7 T_c
            logT = np.log10(np.ravel(T))
            logT[~np.isfinite(logT)] = 3.
            return 10**ufunclike(_Cooling,logT)
        self._CoolingInt=_CoolingInt; self._Cooling = _Cooling; self.Cooling = Cooling

        # Set up ambient gas density/pressure
        cosm = cosmology.cosmology(Om=Om,Ol=Ol,h=h); self.cosm = cosm
        nfw = NFW.NFW(Mvir,zvir,c,cosm); self.nfw = nfw
        self.c = nfw.c
        Rvir = nfw.Rvir * nfw.kpc
        Tion = T0 #K
        rho_flat = n_flat*nfw.mbar
        if useiso:
            def n_iso(r): #r in cm
                return 10.**(-2*np.log10(r/nfw.kpc) - 3) * nfw.Tvir/1211. #scale by CM14 Tvir
            r_iso = 10**(-(np.log10(n_flat*1211./nfw.Tvir)+3)/2.)*nfw.kpc #cm
            def _rho0(r): #r in cm
                if r < r_iso: return rho_flat
                return n_iso(r)*nfw.mbar
            self.r_iso = r_iso; self.n_iso = n_iso
        elif useflat:
            def _rho0(r):
                return rho_flat
        elif usechampagne:
            import champagne
            tau_1Myr = 1.e6 * 3.16e7
            kB = 1.38e-16
            cion = np.sqrt(kB*Tion/(0.59 * nfw.mp))
            cdat = np.load('champagne.npy')
            c_x = cdat[:,0]; c_a = cdat[:,1]
            c_narr = champagne.n_(c_a,tau_1Myr)
            tau_flat = np.sqrt(c_narr[0]/n_flat) * tau_1Myr
            print n_flat,tau_flat/tau_1Myr
            c_rarr = c_x * cion * tau_flat
            c_narr = champagne.n_(c_a,tau_flat)
            # extend down to r=0 at nflat
            c_rarr = np.concatenate([[0],c_rarr])
            c_narr = np.concatenate([[c_narr[0]],c_narr])
            c_interp = interpolate.interp1d(c_rarr,c_narr)
            c_rmax = c_rarr[-1]

            rhoIGM = cosm.rho_m(zvir)/6.
            nIGM = rhoIGM/nfw.mbar
            rIGM = 10**((np.log10(nIGM*1211./nfw.Tvir)+3.)/-2.)*nfw.kpc
            self.rhoIGM = rhoIGM; self.nIGM = nIGM; self.rIGM = rIGM

            def n_iso(r): #r in cm
                return 10.**(-2*np.log10(r/nfw.kpc) - 3) * nfw.Tvir/1211. #scale by CM14 Tvir
            def _rho0(r):
                if r < c_rmax: return c_interp(r)*0.59*nfw.mp
                if r > rIGM: return rhoIGM
                return n_iso(r)*0.59*nfw.mp
            self.c_interp = c_interp; self.c_rmax = c_rmax
            self.n_iso = n_iso
        else:
            rhoIGM = cosm.rho_m(zvir)/6.
            def _rho0(r): #r in cm
                if r < Rvir: return rho_flat
                return (rho_flat-rhoIGM)*np.exp(-(r-Rvir)/(1.2*Rvir))+rhoIGM
            self.rhoIGM = rhoIGM
        def rho0(r):
            return ufunclike(_rho0,r)
        self.Rvir = Rvir; self.Tion = Tion
        self.rho_flat = rho_flat; self.n_flat = n_flat
        self.useiso = useiso
        self.usechampagne = usechampagne

        _rarr = np.arange(0,300*nfw.Rvir,.001)[1:]*nfw.kpc
        _rhoarr = rho0(_rarr)
        _rho0interp = interpolate.UnivariateSpline(_rarr, _rhoarr, s=0)
        _rho0pinterp= _rho0interp.derivative()
        _rho0parr = _rho0pinterp(_rarr)
        _rho0ratio = _rho0parr/_rhoarr
        _rho0ratiointerp = interpolate.UnivariateSpline(_rarr, _rho0ratio, s=0)
        def rho0ratio(r):
            return _rho0ratiointerp(r)
        #minrho0p = np.abs(np.min(np.diff(_rhoarr))/np.min(np.diff(_rarr)))
        #def _rho0p(r):
        #    out = np.abs(_rho0pinterp(r))
        #    if out < minrho0p: return 0
        #    return out
        #def rho0p(r): #derivative of rho0
        #    return ufunclike(_rho0p,r)
        if useiso:
            def _P0(r):
                if r < r_iso: return rho0(r)*nfw.kB*Tion/nfw.mbar    #ionized
                else: return rho0(r)*nfw.kB*nfw.Tvir/(1.23 * nfw.mp) #neutral
            def P0(r):
                return ufunclike(_P0,r)
            self.cvir = np.sqrt(nfw.kB * nfw.Tvir / (1.23 * nfw.mp)) #cm/s
            self._P0 = _P0; self.P0 = P0
        elif useflat:
            def P0(r):
                return rho0(r)*nfw.kB*Tion/nfw.mbar
            self.cvir = np.sqrt(nfw.kB * Tion / (1.23 * nfw.mp)) #cm/s
            self.P0 = P0
        elif usechampagne:
            def _P0(r):
                return rho0(r)*nfw.kB*Tion/nfw.mbar
                #if r < c_rmax: return rho0(r)*nfw.kB*Tion/nfw.mbar    #ionized
                #else: return rho0(r)*nfw.kB*nfw.Tvir/(1.23 * nfw.mp) #neutral
            def P0(r):
                return ufunclike(_P0,r)
            self.cvir = np.sqrt(nfw.kB * nfw.Tvir / (1.23 * nfw.mp)) #cm/s
            self._P0 = _P0; self.P0 = P0
        else:
            def P0(r): #cgs: erg/cm^3
                return rho0(r)*nfw.kB*Tion/nfw.mbar
            self.P0 = P0
        self._rarr = _rarr; self._rhoarr = _rhoarr
        self._rho0interp = _rho0interp; self._rho0pinterp = _rho0pinterp
        self._rho0 = _rho0; self.rho0 = rho0
        #self._rho0p = _rho0p; self.rho0p = rho0p
        self._rho0parr = _rho0parr
        self._rho0ratio = _rho0ratio
        self._rho0ratiointerp = _rho0ratiointerp
        self.rho0ratio = rho0ratio

        def Pb(E,R):
            return E/(2*np.pi*R**3)
        def nbar(R,E,T):
            return 5./3 * self.Pb(E,R)/(self.nfw.kB * T) #5/3 n_c
        def vdot(y,t): #Rdotdot
            v,R,E,T = y
            return 3*(Pb(E,R)-P0(R))/(R*rho0(R)) - 3*v*v/R - v*v*rho0ratio(R) - nfw.Gc*nfw.Msun*nfw.Mltr(R/nfw.kpc)/(R**2)
        #def vdot(y,t): #if only including gravity from uniform gas, for useflat
        #    v,R,E,T = y
        #    return 3*(Pb(E,R)-P0(R))/(R*rho0(R)) - 3*v*v/R - v*v*rho0p(R)/rho0(R) - 4*np.pi*nfw.Gc*n_flat*1.23*nfw.mp*R/3
        def Rdot(y,t):
            return y[0]
        def Edot(y,t):
            v,R,E,T = y
            return -4*np.pi*R*R*Pb(E,R)*v +L(t) - (4*np.pi*R**3/3)*(nbar(R,E,T)**2)*Cooling(T)
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
            if y[0] <= 0: return [0,0,0,0]
            return [vdot(y,t),Rdot(y,t),Edot(y,t),Tdot(y,t)]
        self.eta = eta; self.C1 = C1; self.C2 = C2
        self.Pb = Pb; self.nbar = nbar
        self.vdot = vdot; self.Rdot = Rdot; self.Edot = Edot
        self.Tdot = Tdot; self.Pdot = Pdot
        self.ydot = ydot

    def run_model(self,Ei,tmax=100,dt=.005):
        nfw = self.nfw
        st = sedov.sedov(self.rho_flat,Ei)
        ti = 10000.*3e7 #10^4 years to seconds
        Ri = st.Rs(ti); vi = st.us(ti)
        Ti = self.Pb(Ei,Ri)/(nfw.kB * (self.rho_flat/nfw.mbar))
        #if Ti > 10**8.5: 
            #print "WARNING: Ti = {0:.2e}, setting to 1e8.5".format(Ti)
            #Ti = 10**8.5
            #pass

        tmin = ti/(3e7*1e6)
        tarr = np.arange(tmin,tmax+dt,dt)*(3e7 * 1e6)
        # setup initial conditions
        y0 = [vi,Ri,Ei,Ti] #cgs units
        
        # integration
        yout,out = integrate.odeint(self.ydot, y0, tarr,full_output=True)

        tarr = tarr/(3e7 * 1e6)
        varr = yout[:,0]/1e5 
        Rarr = yout[:,1]/nfw.kpc
        Earr = yout[:,2]/1e51
        Tarr = yout[:,3]
        if not self.useiso and not self.usechampagne:
            ii = varr <= 0
        else:
            ii = varr <= self.cvir/ 1e5

        try:
            iimax = np.min(np.where(ii)[0])
        except ValueError:
            iimax = len(ii)
        ii = np.zeros(len(ii)).astype(bool)
        ii[0:iimax] = True



        tarr = tarr[ii]; varr = varr[ii]; Rarr = Rarr[ii]; Earr = Earr[ii]; Tarr = Tarr[ii]
        self.R_SN[Ei] = np.nanmax(Rarr); self.t_SN[Ei] = np.nanmax(tarr)
        return tarr,varr,Rarr,Earr,Tarr

    def plot_model(self,tarr,varr,Rarr,Earr,Tarr,fig=None,**kwargs):
        if fig==None:
            fig,axarr = plt.subplots(3,2,figsize=(8,8))
        else:
            axarr = np.reshape(fig.axes,(3,2))

        l, = axarr[0,0].plot(tarr,Rarr,**kwargs)
        axarr[0,0].set_xlabel('t (Myr)')
        axarr[0,0].set_ylabel('R (kpc)')
        axarr[0,1].plot(tarr,varr,**kwargs)
        axarr[0,1].plot(tarr,[self.cvir/1.e5 for t in tarr],'k:')
        axarr[0,1].set_xlabel('t (Myr)')
        axarr[0,1].set_ylabel('v (km/s)')
        axarr[0,1].set_yscale('log')
        axarr[1,0].plot(tarr,Earr,**kwargs)
        axarr[1,0].set_xlabel('t (Myr)')
        axarr[1,0].set_ylabel('E_51')
        axarr[1,0].set_yscale('log')
        axarr[1,1].plot(tarr,Tarr,**kwargs)
        axarr[1,1].set_xlabel('t (Myr)')
        axarr[1,1].set_ylabel('T [k]')
        axarr[1,1].set_yscale('log')

        axarr[2,0].plot(tarr,self.rho0(Rarr*self.nfw.kpc),**kwargs)
        axarr[2,0].set_xlabel('t (Myr)')
        axarr[2,0].set_ylabel('rho0 (g/cc)')
        axarr[2,0].set_yscale('log')
        axarr[2,1].plot(tarr,self.Pb(Earr*1e51,Rarr*self.nfw.kpc),**kwargs)
        axarr[2,1].plot(tarr,self.P0(Rarr*self.nfw.kpc),**kwargs)
        axarr[2,1].set_xlabel('t (Myr)')
        axarr[2,1].set_ylabel('Pb,P0 (erg/cc)')
        axarr[2,1].set_yscale('log')
        return l

if __name__=="__main__":
    zvir = 25
    Mvir = 1e6
    #zvir = 20
    #Mvir = 5.e5
    ESNarr = 10**np.array([50,50.5,51,51.5,52,52.5,53])
    nflatarr = 10**np.array([0,-.5,-1,-1.5,-2])
    #nflatarr = 10**np.array([-.5])
    modelarr = [MFR01(Mvir,zvir,n_flat=n_flat,useiso=False,useflat=False,usechampagne=True) for n_flat in nflatarr]

    for model in modelarr:
        for Ei in ESNarr:
            model.run_model(Ei,tmax=5000)
    
    nfw = modelarr[0].nfw

    fig,ax = plt.subplots()
    output = []
    for nflat,model in zip(nflatarr,modelarr):
        Rarr = [model.R_SN[Ei] for Ei in ESNarr]
        ax.plot(ESNarr,Rarr,'o-',label=r'$\log n_{flat}$='+str(round(np.log10(nflat),1)))
        output.append([ESNarr,Rarr])
    ax.plot([1e50,1e53],[nfw.Rvir,nfw.Rvir],'k:')
    ax.set_xscale('log')
    ax.set_xlabel('E (erg)'); ax.set_ylabel('R (kpc)')
    ax.legend(loc='upper left')
    #plt.show()
    with open('MFRchampagne_ESNarr_Rarr.p','w') as f: pickle.dump(output,f)

    fig,ax = plt.subplots()
    output = []
    for nflat,model in zip(nflatarr,modelarr):
        tarr = [model.t_SN[Ei] for Ei in ESNarr]
        ax.plot(ESNarr,tarr,'o-',label=r'$\log n_{flat}$='+str(round(np.log10(nflat),1)))
        output.append([ESNarr,tarr])
    ax.set_xscale('log')
    ax.set_xlabel('E (erg)'); ax.set_ylabel('t (Myr)')
    ax.legend(loc='upper left')
    #plt.show()
    with open('MFRchampagne_ESNarr_tarr.p','w') as f: pickle.dump(output,f)

    fig,ax = plt.subplots()
    output = []
    for nflat,model in zip(nflatarr,modelarr):
        plotR = np.logspace(-3,1.5,200)
        plotn = model.rho0(plotR*model.nfw.kpc)/model.nfw.mbar
        ax.plot(plotR,plotn,label=r'$\log n_{flat}$='+str(round(np.log10(nflat),1)))
        output.append([plotR,plotn])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('R (kpc)'); ax.set_ylabel(r'n (cm$^{-3}$)')
    ax.set_xlim((10**-3,10**1.5)); ax.set_ylim((10**-4,10**0.5))
    ax.legend(loc='upper right')
    with open('MFRchampagne_Rarr_narr.p','w') as f: pickle.dump(output,f)
    plt.show()

def plot_flat_stuff():
    zvir = 10
    Mvir = 1e8
    ESN = 10.**51
    T0arr = [1e3, 1e4]
    nflatarr = 10**np.array([0,-.5,-1,-1.5,-2])
    modelarr = [[MFR01(Mvir,zvir,n_flat=n_flat,T0=T0,useiso=False,useflat=True) for n_flat in nflatarr] for T0 in T0arr]

    for nmodel in modelarr:
        for model in nmodel:
            tarr,varr,Rarr,Earr,Tarr = model.run_model(ESN)
    allR = [[model.R_SN[ESN] for model in nmodel] for nmodel in modelarr]
    

    tauarr = np.array([10,70,100])
    Dttauarr = 8.1e-4 * tauarr

    fig,axarr = plt.subplots(2,1,figsize=(8,10))
    ax = axarr[0]
    for i,thisR in enumerate(allR):
        ax.plot(nflatarr,thisR,'o-',label='T={0:.0f}K'.format(T0arr[i]))
    ax.set_xscale('log')
    ax.set_xlabel(r'n (cm$^{-3}$)')
    ax.set_ylabel('R (kpc)')
    ax.legend(loc='upper right')

    ax = axarr[1]

    colorarr = ['b','g']
    stylearr = ['--',':','-.']
    for i,thisR in enumerate(allR):
        thisR = np.array(thisR)
        Vmixarr = 4*np.pi/3 * (thisR*model.nfw.kpc)**3
        massarr = Vmixarr * nflatarr * model.nfw.mp/(model.nfw.Msun)
        ax.plot(nflatarr,massarr,color=colorarr[i])

        turbmixVlist = [4*np.pi/3 * ((thisR*model.nfw.kpc)**2 + 6*Dttau*(model.nfw.kpc)**2)**1.5 for Dttau in Dttauarr]
        for j,turbmixV in enumerate(turbmixVlist):
            massarr = np.array(turbmixV) * nflatarr * model.nfw.mp/(model.nfw.Msun)
            if i==0:
                ax.plot(nflatarr,massarr,stylearr[j],color=colorarr[i],label=r'$\tau={0}\ Myr$'.format(tauarr[j]))
            else:
                ax.plot(nflatarr,massarr,stylearr[j],color=colorarr[i])

    ax.legend(loc='upper left')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'n (cm$^{-3}$)')
    ax.set_ylabel(r'$M_{mix}$ ($M_\odot$)')
    plt.show()

def old_main_model():
    zvir = 25
    Mvir = 1e6
    ESNarr = 10**np.array([50,50.5,51,51.5,52,52.5,53])
    nflatarr = 10**np.array([0,-.5,-1,-1.5,-2])
    modelarr = [MFR01(Mvir,zvir,n_flat=n_flat,useiso=True) for n_flat in nflatarr]

    for model in modelarr:
        for Ei in ESNarr:
            model.run_model(Ei,tmax=200)
    
    nfw = modelarr[0].nfw

    fig,ax = plt.subplots()
    for nflat,model in zip(nflatarr,modelarr):
        Rarr = [model.R_SN[Ei] for Ei in ESNarr]
        ax.plot(ESNarr,Rarr,'o-',label=r'$\log n_{flat}$='+str(round(np.log10(nflat),1)))
    ax.plot([1e50,1e53],[nfw.Rvir,nfw.Rvir],'k:')
    ax.set_xscale('log')
    ax.set_xlabel('E (erg)'); ax.set_ylabel('R (kpc)')
    ax.legend(loc='upper left')
    plt.show()

    fig,ax = plt.subplots()
    for nflat,model in zip(nflatarr,modelarr):
        tarr = [model.t_SN[Ei] for Ei in ESNarr]
        ax.plot(ESNarr,tarr,'o-',label=r'$\log n_{flat}$='+str(round(np.log10(nflat),1)))
    ax.set_xscale('log')
    ax.set_xlabel('E (erg)'); ax.set_ylabel('t (Myr)')
    ax.legend(loc='upper left')
    plt.show()

    fig,ax = plt.subplots()
    for nflat,model in zip(nflatarr,modelarr):
        plotR = np.logspace(-3,1.5,200)
        plotn = model.rho0(plotR*model.nfw.kpc)/model.nfw.mbar
        ax.plot(plotR,plotn,label=r'$\log n_{flat}$='+str(round(np.log10(nflat),1)))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('R (kpc)'); ax.set_ylabel(r'n (cm$^{-3}$)')
    ax.set_xlim((10**-3,10**1.5)); ax.set_ylim((10**-4,10**0.5))
    ax.legend(loc='upper right')
    plt.show()

def old2():
    zvirarr = [30,25,20,15]
    Mvir = 1e6
    Earr = 10**np.array([50,50.5,51,51.5,52,52.5])
    modelarr = [MFR01(Mvir,zvir) for zvir in zvirarr]
    for model in modelarr:
        for Ei in Earr:
            model.run_model(Ei,tmax=150)
    
    fig,ax = plt.subplots()
    for zvir,model in zip(zvirarr,modelarr):
        Rarr = [model.R_SN[Ei] for Ei in Earr]
        ax.plot(Earr,Rarr,'o-',label='z='+str(zvir))
    ax.set_xscale('log')
    ax.set_xlabel('E (erg)'); ax.set_ylabel('R (kpc)')
    ax.legend(loc='upper left')
    plt.show()

def old():
    fig,axes = plt.subplots(2,2,figsize=(10,10))
    fig.subplots_adjust(wspace=.25)
    model = MFR01(1e6, 25)
    tarr,varr,Rarr,Earr,Tarr = model.run_model(1e51)
    model.plot_model(tarr,varr,Rarr,Earr,Tarr,fig=fig)
    tarr,varr,Rarr,Earr,Tarr = model.run_model(1e52)
    model.plot_model(tarr,varr,Rarr,Earr,Tarr,fig=fig)
    ax = axes[1,1]; ax.legend(['1e51','1e52'],loc='upper right')
    plt.show()
