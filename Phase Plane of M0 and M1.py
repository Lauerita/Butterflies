import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

####### Stage 1 ########

# Parameters
alpha = 2.232  # reproductive rate 
gamma = 0.03571  # maturation rate 
mu1 = 0.11495
mu2 = 0.07143
mu0 = 0.15975
M0_initial = 150000000
I10 = 0
M10 = 0
I20 = 0
M20 = 0

# ODE system definition
def butterfly(X, t):
    M0, I1, M1, I2, M2 = X
    dM0dt = -mu0 * M0  # The change in M0 (adults gen0)
    dI1dt = M0 * alpha - I1 * mu1 - gamma * I1  # The change in I1 (larvae gen1)
    dM1dt = I1 * gamma - M1 * mu2  # The change in M1 (adults gen1)
    dI2dt = M1 * alpha - I2 * mu1 - gamma * I2  # The change in I2 (larvae gen2)
    dM2dt = I2 * gamma - M2 * mu2  # The change in M2 (adults gen2)
    
    return np.array([dM0dt, dI1dt, dM1dt, dI2dt, dM2dt])

# Time range for integration
days = 5 * 30
month = 3 * 30
t = np.linspace(month, days + month, days + 1)

# Initial conditions
X0 = M0_initial, I10, M10, I20, M20

# Solve the ODE system
res = integrate.odeint(butterfly, X0, t)

# Extract solutions for M0, I1, M1
M0, I1, M1, I2, M2 = res.T

# Create a grid of M0 and M1 values for the stream plot
M0_grid, M1_grid = np.meshgrid(np.linspace(min(M0), max(M0), 20), np.linspace(min(M1), max(M1), 20))

# Initialize derivative grids with the same shape as the meshgrid
dM0dt_grid = np.zeros_like(M0_grid)
dM1dt_grid = np.zeros_like(M1_grid)

# Calculate the derivatives at each point on the grid
for i in range(len(t)):
    # Broadcast M0 value for the time step i over the grid
    I1_value = I1[i]
    # Calculate the derivatives for this M0 value and the current grid of M0 and M1
    dM0dt_grid = -mu0 * M0_grid  # The change in M0 (adults gen0)
    dM1dt_grid = I1_value * gamma - M1_grid * mu2  # The change in M1 (adults gen1)

# Stream plot
plt.figure(figsize=(8, 6))
plt.streamplot(M0_grid, M1_grid, dM0dt_grid, dM1dt_grid, color='b', linewidth=1)

# Labels and title
plt.title('Stream Plot of adult gen1 and adult gen0')
plt.xlabel('Adults Gen 0 (M0)')
plt.ylabel('Adults Gen 1 (M1)')
plt.grid()

# Show plot
plt.savefig('streamplotM1&M0.png', dpi = 300)
plt.show()


