import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sympy
from sympy import *

class PK_N_compartment:

    def __init__(self, a, f, params):
        # set up PK object with a matrix, f forcing function, and parameters

        # determine number of compartments and number of parameters
        self.NC = a.shape[0]
        self.NP = len(params)
        self.order = self.NC*(self.NP + 1)
        self.params = params

        # define symbolic derivatives of a matrix with respect to each param
        ai = {}
        for i, param in enumerate(params):
            ai[i] = diff(a, param)

        # create symoblic matrix A
        A = Matrix(np.zeros([self.order, self.order]))

        for i in range(self.NP+1):
            A[i*self.NC:i*self.NC+self.NC,i*self.NC:i*self.NC+self.NC] = a

        for i in range(1,self.NP+1):
            A[i*self.NC:i*self.NC+self.NC,:self.NC] = ai[i-1]

        # define symbolic forcing function f
        F = Matrix(np.zeros(self.order))

        # populate F with given entries for model forcing function
        for i in range(self.NC):
            F[i] = f[i]

        for i,p in zip(np.arange(self.NC, self.order, self.NC), range(self.NP)):
            for j in range(self.NC):
                F[i+j] = diff(f, params[p])[j]

        self.A = A
        self.F = F

    def set_mle(self, mle_vals):
        # introduce maximum likelihood estimate parameter values to model
        A = self.A
        F = self.F

        subs = {param:mle_val for param,mle_val in zip(self.params, mle_vals)}
        A = A.evalf(subs=subs)
        F = F.evalf(subs=subs)
        A = np.array(A)
        F = np.array(F)
        # not sure why I can't do this in one step..
        self.A = np.array(A, np.float)
        self.F = np.array(F, np.float)

    def set_c0(self, c0):
        self.c0 = np.zeros(self.order)
        self.c0[:self.NC] = c0

    def get_eqn(self):
        return self.A, self.F

    def equations(self, c, t):
        return np.dot(self.A, c) + np.hstack(self.F)

    def solve(self, t_array):
        self.soln = odeint(self.equations, self.c0, t_array)
        return self.soln

    def plot(self, t_array, compartments=False, show=True, label= None, title=None):

        if not compartments:
            compartments = range(self.NC)

        for C in compartments:
            plt.plot(t_array, self.soln[:,C-1], label=label)
        plt.xlabel('time')
        plt.ylabel('concentration')
        if title:
            plt.title(title)
        else:
            plt.title('Concentration in Compartment {}'.format(compartments))
        plt.legend()
        if show:
            plt.show()
        else:
            return plt

    def FIM(self, t_array):
        self.t_array = t_array
        self.NS = len(t_array)
        self.solve(t_array)
        NC = self.NC
        # eqn 14.7: SVi = (dy(t1)/dpi,...,dy(tn)/dpi)'
        # eqn 14.8: SM = (SV1,...,SVl)
        SM = self.soln[:, NC:]
        # define covariance matrix of measurement noise
        E = np.eye(self.NS)
        # eqn 14.20: FIM(0)ij = E(SVi).T * CM^-1 * E(SVj)
        FIM = np.zeros([self.NP, self.NP])
        for i in np.arange(0, self.NP*NC, NC):
            for j in np.arange(0, self.NP*NC, NC):
                for k in range(NC):
                    FIM[i//NC,j//NC] += np.dot(np.dot(SM[:,i+k].T, E), SM[:,j+k])
        FIM*=self.NS 

        return FIM
