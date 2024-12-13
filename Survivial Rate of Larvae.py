import numpy as np
import matplotlib.pyplot as plt

### This shows the survival rate of the larvae as a function of the temperature 
### We assumed all larvae are in stage 3

days = 365
month = 1
t = np.linspace(month, days+month, 365)


plt.figure()

def temp(t, T0, e = -1.102, f = 0.917):
    return T0 * (1 + e * np.sin(2 * np.pi * t / 365 + f))

def survive(T, S = 3):
    y = 1 - (-0.003*T**2 + 0.1411*T - 0.67 - 0.0911*S)
    return y

#plt.plot(temp(t), survive(temp(t)))

T0 = [11, 15, 17, 20]

for i in T0:
    plt.plot(t, survive(temp(t, T0 = i)), label=f'T0 = {i}')
plt.title(' Mortality Rate of the Larvae as a Function of Temperature ')
plt.xlabel('t (days)')
plt.ylabel('Mortality rate')

plt.legend()
plt.grid(True)

plt.savefig('Mortality Rate of Larvae.png', dpi = 300)

    

