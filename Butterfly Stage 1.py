import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

####### Stage 1 ########

# From March to June
days = 3*30
month = 3*30
t = np.linspace(month, days + month, days + 1) 

alpha = 2.232  # reproductive rate 
gamma = 0.03571  # maturation rate 
mu1 = 0.11495
mu2 = 0.07143
mu0 = 0.15975
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

X0 = M0, I10, M10, I20, M20

res = integrate.odeint(butterfly, X0, t)

M0, I1, M1, I2, M2 = res.T

# Assign color scheme: Adults with warm colors, larvae with cool colors
adult_colors = ['b', 'darkred', 'r']
larvae_colors = ['limegreen', 'darkgreen']

plt.figure()
plt.grid()
plt.title("General Graphical Solution of Stage 1")

# Plot adult generation curves with warm colors
plt.plot(t, M0, label='Adult gen0', color=adult_colors[0])
plt.plot(t, M1, label='Adult gen1', color=adult_colors[1])
plt.plot(t, M2, label='Adult gen2', color=adult_colors[2])
#plt.plot(t, M3, label='Adult gen3', color=adult_colors[3])

# Plot larvae generation curves with cool colors
plt.plot(t, I1, label='Larvea gen1', color=larvae_colors[0])
plt.plot(t, I2, label='Larvea gen2', color=larvae_colors[1])
#plt.plot(t, I3, label='Larvea gen3', color=larvae_colors[2])

# Set legend font size
plt.legend(fontsize=8)
plt.xlabel('t (days)')
plt.ylabel('Number of larvae/butterflies')
plt.savefig('Stage1sol.png', dpi = 300)
plt.show()







