import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

T0_values = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

fig, ax = plt.subplots(1, 2)

# Lists to store the results for plotting
M_s_values = []
A_values = []

for i in T0_values:
    days = 30 * 3
    month = 6*30 + 1
    t = np.linspace(month, days+month, days + 1)


    def temp(t, T0, e = -1.102, f = 0.917):
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
        dI_in = -(gamma + survive(temp(t, i)))*I_in
        dM_in = -mu2*M_in + gamma*I_in
        dI_s = alpha*M_in*A - (gamma + survive(temp(t, i)))*I_s
        dM_s = gamma*I_s - mu3*M_s
        dA = growth(temp(t, i))*A*(1 - (A/K)) - A*beta*I_s

            

        return np.array([dI_in, dM_in, dI_s, dM_s, dA])

    X0 = I_in0, M_in0, I_s0, M_s0, A_0

    res = integrate.odeint(butterfly, X0, t)

    I_in, M_in, I_s, M_s, A = res.T
    
    M_s_values.append(M_s[-1])  # Store the last M_s value for plotting
    A_values.append(A[-1])  # Store the last A value for plotting
    
# Plotting the connected scatter points
ax[0].plot(T0_values, M_s_values, marker='o', linestyle='-', color='r')
ax[0].set_xlabel('Value of T0')
ax[0].set_ylabel('Adult monarch population at the end of the season')

ax[1].plot(T0_values, A_values, marker='o', linestyle='-', color='g')
ax[1].set_xlabel('Value of T0')
ax[1].set_ylabel('Unit of milkweed at the end of the season')

plt.tight_layout()
plt.savefig('Effect of Temp on Diapause Butterfly & Milkweed.png', dpi= 300)
plt.show()

