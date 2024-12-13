import numpy as np 
import matplotlib.pyplot as plt

days = 365
month = 1
t = np.linspace(month, days + month, 365)

# Define the growth function with checks for negative and NaN values
def growth(t, R_max=0.007, T_max=35, T_opt=27):
    r = R_max * (T_max - t) /(T_max - T_opt) * (t / T_opt)**(T_opt / (T_max - T_opt))
    r = np.nan_to_num(r, nan=0)  # Replace NaN with 0
    r = np.maximum(r, 0)  # Ensure that r is never less than 0
    return r

# Define the temperature function
def temp(t, T0, e=-1.102, f=0.917):
    y = T0 * (1 + e * np.sin(2 * np.pi * t / 365 + f))
    return y

# Create the figure with a 2x2 grid of subplots
plt.figure()

T0 = [11, 15, 17, 20]

for i in T0:
    growth_rate = growth(temp(t, T0=i))
    plt.plot(t, growth_rate, label=f'T0 = {i}')
    print(growth_rate)

plt.title('Milkweed Growth Rate as Function of Temperature')
plt.xlabel('t (days)')
plt.ylabel('Growth rate (/day)')

# Set the legend title to "Value of T0"
plt.legend()
plt.grid(True)
plt.savefig('milkweed growth rate.png', dpi = 300)
plt.show()







