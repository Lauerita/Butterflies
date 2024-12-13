import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

T0 = [11, 12, 13, 15, 20]

plt.figure()

def temp(t, T0, e = -1.102, f = 0.917):
       y = T0 * (1 + e * np.sin(2 * np.pi * t / 365 + f))
       return y

def growth(t, R_max = 0.07, T_max = 35, T_opt = 27):
       asr = R_max * (T_max - t)/(T_max - T_opt) * (t/T_opt)**(T_opt/(T_max - T_opt))
       return asr

for k in T0:
 
    a = 365
    b = 1
    t = np.linspace(b, a, 50)
    T0 = k
    plt.plot(t, temp(t, k), marker = 'o')

    


fig, ax = plt.subplots(1,2)

for i in T0:
    days = 60
    month = 6*30
    t = np.linspace(month, days+month, days + 1)


    alpha = 2.6 # reproductive rate 
    gamma = 0.03571 # maturation rate 
    mu1 = 0.11495
    mu2 = 0.07143
    mu3 = 0.005
    T0 = i


    K = 1.79188
    beta = 5*10**-9


    I_in0 = 3283396409
    M_in0 = 3114788529
    I_s0 = 0
    M_s0 = 0
    A_0 = 1 # initial unit of milkweed
     


    def butterfly(X, t):
       I_in, M_in, I_s, M_s, A = X 
       dI_in = -(gamma + mu1)*I_in
       dM_in = -mu2*M_in + gamma*I_in
       dI_s = alpha*M_in*A - (gamma + mu1)*I_s
       dM_s = gamma*I_s - mu3*M_s
       dA = growth(temp(t, i))*A*(1 - (A/K)) - A*beta*I_s

           

       return np.array([dI_in, dM_in, dI_s, dM_s, dA])

    X0 = I_in0, M_in0, I_s0, M_s0, A_0

    res = integrate.odeint(butterfly, X0, t)

    I_in, M_in, I_s, M_s, A = res.T
    
    ax[0].scatter(i, M_s[-1])
    ax[1].scatter(i, A[-1])

    
    
    
   

   
   



