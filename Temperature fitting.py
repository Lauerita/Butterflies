import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the model function (adjusted for months)
def model(t, T0, e, f):
    return T0 * (1 + e * np.sin(2 * np.pi * t / 12 + f))

# Given temperature data for each month of the year
t = np.linspace(1, 12, 12)  # months from 1 to 12
y = [30, 31, 39, 51, 63, 72, 77, 76, 68, 56, 45, 35]  # average temperature for each month



# Chnange farenheit into celcius 
for i in range(len(y)):
    y[i] = (5/9)*(y[i] - 32)*1.1

# Initial guesses for the parameters T0, e, f
# T0 is roughly the average temperature, e is half the range, and f is 0 to start
T0_guess = np.mean(y)
e_guess = (np.max(y) - np.min(y)) / 2
f_guess = 0  # initial guess for phase shift

# Perform the curve fitting
initial_guess = [T0_guess, e_guess, f_guess]
popt, pcov = curve_fit(model, t, y, p0=initial_guess)

# Extract the optimized parameters
T0, e, f = popt
print(f"Optimized parameters: T0 = {T0}, e = {e}, f = {f}")

# Generate the fitted temperature values
y_fitted = model(t, *popt)

# Plot the data and the fitted model
plt.figure(figsize=(8, 5))
plt.plot(t, y, 'bo-', label='Observed Data')
plt.plot(t, y_fitted, 'r-', label='Fitted Model')
plt.xlabel('Month of the Year')
plt.ylabel('Temperature (°C)')
plt.title('Monthly Temperature Fit')
plt.legend()
plt.xticks(t)
plt.grid(True)

plt.savefig('temperature curve.png', dpi = 300)

plt.show()
