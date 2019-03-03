import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
import sympy
from sympy import *
import FIM_PK
from FIM_PK import PK_N_compartment

k_12, k_el, V1, V2 = symbols(('k_12', 'k_el', 'V1', 'V2'), positive=True, real=True)

# time dependent mass balance equations
# (dC1/dt) = -k12*C1
# (dC2/dt) = k12*V1/V2*C1 - kel*C2

# define coefficient matrix a
a  = Matrix([[-k_12, 0],
             [k_12*V1/V2, -k_el]])
# no forcing function
f = Matrix([0,0])

# define params used in model
params = (k_12, k_el, V1, V2)

# create my PK object
PKfim = PK_N_compartment(a, f, params)
As, Fs = PKfim.get_eqn()

# set mle vals for drugs 1, 2, and 3
mle_vals = ((.2, .1, 1, 12), (.2, .5, 5, 4), (.05, 1.5, .2, 1))
days = 3

#%% plot drug profiles
t_array = np.linspace(0, days*24, 100)
for drug, mle_val in enumerate(mle_vals):
    PKfim = PK_N_compartment(a, f, params)
    PKfim.set_mle(mle_val)

    # set initial condition
    c0 = np.array([.5/mle_val[2],0])
    PKfim.set_c0(c0)

    PKfim.solve(t_array)
    plt = PKfim.plot(t_array, label='Drug {}'.format(drug+1), compartments=[2], show=False)

plt.ylabel('Drug Concentration (mg/L)')
plt.xlabel('time (hours)')
plt.legend()
plt.show()

#%% determine optimal timescale to sample data
time = np.linspace(0, days*24, 1000)
N_scales = 50
scales = np.logspace(-3, .05, N_scales)
tspan = [max(time)*scale for scale in scales]
np.savetxt('FIM_tspan.csv', tspan)
D_optimality = np.zeros(N_scales)
A_optimality = np.zeros(N_scales)
E_optimality = np.zeros(N_scales)
best_time_scale = {}

for drug, mle_val in enumerate(mle_vals):
    PKfim = PK_N_compartment(a, f, params)
    PKfim.set_mle(mle_val)

    # set initial condition
    c0 = np.array([.5/mle_val[2],0])
    PKfim.set_c0(c0)

    for i, scale in enumerate(scales):
        t_array = time*scale
        FIM = Matrix(PKfim.FIM(t_array)[:2,:2])
        eig, V = np.linalg.eig(np.array(FIM).astype(np.float))
        D_optimality[i] = FIM.det()
        A_optimality[i] = FIM.trace()
        E_optimality[i] = min(eig)

    best_time_scale[drug] = max(time*scales[np.argmax(E_optimality)])
    # save data
    fname = 'drug{}_E_optimality.csv'.format(drug+1)
    np.savetxt(fname, E_optimality)
    plt.plot(tspan, E_optimality/np.max(E_optimality), label=r'Drug {0}, $\tau$: {1:.1f} hrs'.format(drug+1, best_time_scale[drug]))
    #plt.yscale('log')
    plt.ylabel('Normalized E optimality')
    plt.xlabel('Time Frame (hours)')
    plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
