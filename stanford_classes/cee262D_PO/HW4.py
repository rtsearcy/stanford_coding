# -*- coding: utf-8 -*-
# CEE 262D - HW 4
# RTS - 10.28.2019

import numpy as np
import matplotlib.pyplot as plt
# %% Problem 1

T = 12.42 * 60 * 60 # Tidal period (s)
L = 8000 # Length of Elkhorn SLough (m)
H0 = 4 # average depth (m)
dH = 1.5 # change in depth from tide (m)
g = 9.81 # gravity

def H(t):
    return H0 + dH*np.sin((2*np.pi*t)/T)

def H_prime(t):
    return (2*np.pi*dH/T) * np.cos((2*np.pi*t)/T)

def H_double_prime(t):
    return - (4*np.pi*np.pi*dH/(T*T)) * np.sin((2*np.pi*t)/T)

def u(x,t):
    return ((L-x)/H(t)) * H_prime(t)


# %% a.
    
def a(x,t):
    #return (L-x)*(((1/H_t(t)) * H_double_prime(t)) - ((1/(H_t(t)**2))*(H_prime(t)**2)))
    
    dudt = (L-x)*(((1/H(t))*H_double_prime(t)) - ((u(x,t)**2)/(L-x)**2))
    ududx = -(u(x,t)**2)/(L-x)
    
    return dudt + ududx

a1 = a(0,0)
a2 = a(0,T/4)
a3 = a(0,T/2)
a4 = a(0,3*T/4)

#t = np.linspace(0,T,50)
#plt.plot(a(0,t))
#plt.plot(a(4000,t))

# %% b. # non linear accelerations
def dEta(x,t):
    return - a(x,t) / g

slope0 = dEta(4000,0)
slopeT4 = dEta(4000,T/4)
slopeT2 = dEta(4000,T/2)
slope3T4 = dEta(4000,3*T/4)


# %% c
f = 2 * (7.27*10**-5) * np.sin(np.pi*36.8/180)
W = 200
dsig = 0.45*f*W/g

# %% Problem 2
v1 = 1.5
rho1 = 1024
v2 = 0.5
rho2 = 1025.5
f = 6.1*10**-5
g = 9.81
W = 80000

dE = W* f*v1/g # 0.75 m

