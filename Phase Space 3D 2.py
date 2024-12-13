import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

days = 30 * 3
month = 6*30 
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
# Create 3D phase space plot
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D trajectory
ax.plot3D(I_s, M_s, A, color='blue')

# Scatter plot of the points to show progression
ax.scatter3D(I_s[0], M_s[0], A[0], color='green', label='Start')
ax.scatter3D(I_s[-1], M_s[-1], A[-1], color='red', label='End')

# Labeling
ax.set_xlabel('I_s')
ax.set_ylabel('M_s')
ax.set_zlabel('A')
ax.set_title('3D Phase Space Trajectory')

plt.legend()
plt.tight_layout()
plt.savefig('3D Phase space stage 2.png', dpi = 300)
plt.show()