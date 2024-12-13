import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

days = 30 * 3
month = 180
t = np.linspace(month, days+month, days + 1)


def temp(t, T0 = 11.98, e = -1.102, f = 0.917):
    y = T0 * (1 + e * np.sin(2 * np.pi * t / 365 + f))
    return y


def growth(t, R_max = 0.007, T_max = 35, T_opt = 27):
    asr = R_max * (T_max - t)/(T_max - T_opt) * (t/T_opt)**(T_opt/(T_max - T_opt))
    return asr

def survive(T, S = 3):
    y = 1 - (-0.003*T**2 + 0.1411*T - 0.67 - 0.0911*S)
    return y


alpha = 2.6 # reproductive rate 
gamma = 0.03571 # maturation rate 
mu2 = 0.07143
mu3 = 0.005

a = -0.05 # Unknown parameter 

K = 1.79188
beta = 5*10**-9


I_in0 = 75980106.51386052
M_in0 = 149239743.87052593
I_s0 = 0
M_s0 = 0
A_0 = 1 # initial unit of milkweed
  


def butterfly(X, t):
    I_in, M_in, I_s, M_s, A = X 
    dI_in = -(gamma + survive(temp(t)))*I_in
    dM_in = -mu2*M_in + gamma*I_in
    dI_s = alpha*M_in*A - (gamma + survive(temp(t)))*I_s
    dM_s = gamma*I_s - mu3*M_s
    dA = growth(temp(t))*A*(1 - (A/K)) - A*beta*I_s

        

    return np.array([dI_in, dM_in, dI_s, dM_s, dA])

X0 = I_in0, M_in0, I_s0, M_s0, A_0

res = integrate.odeint(butterfly, X0, t)

I_in, M_in, I_s, M_s, A = res.T

# Create phase space plot
plt.figure(figsize=(15, 5))

# M0 vs I1 subplot
plt.subplot(131)
plt.plot(I_s, M_s)
plt.title('Phase Space: I_s vs M_s')
plt.scatter(I_s[0], M_s[0], label = 'start')
plt.scatter(I_s[-1], M_s[-1], label = 'end')
plt.xlabel('I_s')
plt.ylabel('M_s')
plt.legend()

# M0 vs M1 subplot
plt.subplot(132)
plt.plot(I_s, A)
plt.title('Phase Space: I_s vs A')
plt.scatter(I_s[0], A[0], label = 'start')
plt.scatter(I_s[-1], A[-1], label = 'end')
plt.xlabel('I_s')
plt.ylabel('A')
plt.legend()

# I1 vs M1 subplot
plt.subplot(133)
plt.plot(M_s, A)
plt.title('Phase Space: M_s vs A')
plt.scatter(M_s[0], A[0], label = 'start')
plt.scatter(M_s[-1], A[-1], label = 'end')
plt.xlabel('M_s')
plt.ylabel('A')
plt.legend()


plt.tight_layout()
plt.savefig('2d Phase Space stage 2.png', dpi = 300)
plt.show()
