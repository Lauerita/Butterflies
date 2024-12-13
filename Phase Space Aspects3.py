import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define parameters
days = 5*30  # 5 months
month = 3*30  # starting from 3 months
t = np.linspace(month, days + month, days + 1)

# Parameters
alpha = 2.232  # reproductive rate
gamma = 0.03571  # maturation rate
mu1 = 0.11495
mu2 = 0.07143
mu0 = 0.15975

# Initial conditions
M0 = 150000000
I10 = 0
M10 = 0
I20 = 0
M20 = 0

def butterfly(X, t):
    M0, I1, M1, I2, M2 = X
    dM0dt = -mu0*M0
    dI1dt = M0*alpha - I1*mu1 - gamma*I1
    dM1dt = I1*gamma - M1*mu2
    dI2dt = M1*alpha - I2*mu1 - gamma*I2
    dM2dt = I2*gamma - M2*mu2
    return np.array([dM0dt, dI1dt, dM1dt, dI2dt, dM2dt])

# Initial state
X0 = [M0, I10, M10, I20, M20]

# Solve ODE
res = integrate.odeint(butterfly, X0, t)
M0, I1, M1, I2, M2 = res.T

# Create phase space plot
plt.figure(figsize=(15, 5))

# M0 vs I1 subplot
plt.subplot(131)
plt.plot(M0, I1)
plt.title('Phase Space: M0 vs I1')
plt.scatter(M0[0], I1[0], label = 'start')
plt.scatter(M0[-1], I1[-1], label = 'end')
plt.xlabel('M0')
plt.ylabel('I1')
plt.legend()

# M0 vs M1 subplot
plt.subplot(132)
plt.plot(M0, M1)
plt.title('Phase Space: M0 vs M1')
plt.scatter(M0[0],  M1[0], label = 'start')
plt.scatter(M0[-1], M1[-1], label = 'end')
plt.xlabel('M0')
plt.ylabel('M1')
plt.legend()

# I1 vs M1 subplot
plt.subplot(133)
plt.plot(I1, M1)
plt.title('Phase Space: I1 vs M1')
plt.scatter(I1[0], M1[0], label = 'start')
plt.scatter(I1[-1], M1[-1], label = 'end')
plt.xlabel('I1')
plt.ylabel('M1')

plt.legend()
plt.tight_layout()
plt.savefig('2d Phase Space stage 1.png', dpi = 300)
plt.show()
