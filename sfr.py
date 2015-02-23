## TODO

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

