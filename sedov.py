import numpy as np
import asciitable
from scipy.integrate import odeint
from scipy.interpolate import interp1d

class sedov(object):
    def __init__(self,rho0,E):
        self.rho0 = rho0
        self.E = E
        self._load()

    def _load(self):
        filename='sedov.csv'
        data = asciitable.read(filename)
        L = data['col1']
        R = data['col2']
        R[np.abs(R) < 1e-8] = 0
        V = data['col3']
        P = data['col4']
        self.R = interp1d(L,R)
        self.V = interp1d(L,V)
        self.P = interp1d(L,P)
    def Rs(self,t):
        return 1.15*(self.E*t*t/self.rho0)**0.2
    def us(self,t):
        return 1.15*.4*(self.E/self.rho0)**0.2 * t**(-.6)
    def _L(self,r,t):
        return r/self.Rs(t)
    def rho(self,r,t):
        return 4*self.rho0*self.R(self._L(r,t))
    def v(self,r,t):
        return .75*self.us(t)*self.V(self._L(r,t))
    def p(self,r,t):
        return .75*self.rho0*self.us(t)**2 * self.P(self._L(r,t))

#def ydot(y,L):
#    R,V,P = y
#    a = 4*L-3*V
#    b = L*(-5*P + R*a*a)
#
#    Rprime = 6*R*(4*L*L*R*V+9*R*V*V*V-3*L*(2*P+5*R*V*V))/(b*a)
#    Vprime = 2*(-6*L*P+5*P*V-12*L*L*R*V+9*L*R*V*V)/b
#    Pprime = -2*P*R*(24*L*L-23*L*V+15*V*V)/b
#
#    return [Rprime,Vprime,Pprime]
