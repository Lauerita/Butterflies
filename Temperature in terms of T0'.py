import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

T0_values = [11, 13, 15, 17, 19, 20]

# Create a figure for plotting
plt.figure()

# Define the temperature function
def temp(t, T0, e = -1.102, f = 0.917):
    return T0 * (1 + e * np.sin(2 * np.pi * t / 365 + f))

# Define the growth function (not used here, but can be included as needed)
def growth(t, R_max = 0.07, T_max = 35, T_opt = 27):
    asr = R_max * (T_max - t)/(T_max - T_opt) * (t/T_opt)**(T_opt/(T_max - T_opt))
    return asr

# Create a list of colors based on the amplitude of each plot
amplitudes = []
for k in T0_values:
    t = np.linspace(1, 365, 12)
    temp_values = temp(t, k)
    amplitude = np.max(temp_values) - np.min(temp_values)  # Compute amplitude
    amplitudes.append(amplitude)

# Normalize the amplitudes to a color scale (darker = higher amplitude)
norm = plt.Normalize(min(amplitudes), max(amplitudes))
colors = plt.cm.plasma(norm(amplitudes))  # Using cividis colormap for a color range

# Plot each temperature curve with the corresponding color based on its amplitude
for k, color in zip(T0_values, colors):
    t = np.linspace(1, 365, 12)
    plt.plot(t, temp(t, k), marker='o', label=f'T0 = {k}', color=color)

# Add labels and legend
plt.legend()
plt.ylabel('Temperature (°C)')
plt.xlabel('t (days)')
plt.title('How the Parameter T0 Changes the Temperature Curve')
plt.grid(True)

plt.savefig('parameter T0.png', dpi = 300)