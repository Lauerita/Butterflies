import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import integrate

days = 60
month = 6*30
t = np.linspace(month, days+month, days + 1)


def temp(t, T0 = 11.98, e = -1.102, f = 0.917):
    y = T0 * (1 + e * np.sin(2 * np.pi * t / 365 + f))
    return y



a = [-1, -0.5, -0.25, -0.1, -0.05, -0.025, 0]

fig, ax = plt.subplots(1,3)

for i in a:
    
    alpha = 2.6 # reproductive rate 
    gamma = 0.03571 # maturation rate 
    mu1 = 0.11495
    mu2 = 0.07143
    mu3 = 0.005

    K = 1.79188
    beta = 5*10**-9
    a = i

    I_in0 = 3283396409
    M_in0 = 3114788529
    I_s0 = 0
    M_s0 = 0
    A_0 = 1 # initial unit of milkweed
      


    def butterfly(X, t):
        I_in, M_in, I_s, M_s, A = X 
        dI_in = -(gamma + mu1)*I_in
        dM_in = -mu2*M_in + gamma*I_in
        dI_s = alpha*M_in*A - (gamma + mu1)*I_s
        dM_s = gamma*I_s - mu3*M_s
        if A > 0:
            dA = (a*(temp(t) - 15)*(temp(t)- 35)*A*(1 - (A/K)) - A*beta*I_s)
        else:
            dA = 0

        return np.array([dI_in, dM_in, dI_s, dM_s, dA])

    X0 = I_in0, M_in0, I_s0, M_s0, A_0

    res = integrate.odeint(butterfly, X0, t)

    I_in, M_in, I_s, M_s, A = res.T


    ax[0].plot(t, I_s)
    ax[1].plot(t, M_s)
    ax[2].plot(t, A, label = i)
    
ax[0].set_title('Last generation Larvae', fontsize = 7)
ax[1].set_title('Diapaused Butterflies', fontsize = 7)
ax[2].set_title('Unit of Milkweed', fontsize = 7)
plt.legend(fontsize = 5)

for i in range(3):
    ax[i].tick_params(axis = 'both', labelsize = 5)





