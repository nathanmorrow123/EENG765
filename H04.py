#Nathan Morrow AFIT EENG 765 H04

import numpy as np
import util
import matplotlib.pyplot as plt
import math


'''
Case 1: Known starting position and a known heading
sigma_x = 0.1
sigma_y = 0.1
sigma_a = 5*10^(-5)
Case 2: Unknown starting position and known heading
sigma_x = 100
sigma_y = 100
Case 3: Known starting position and an unknown heading
sigma_a = math.pi
Case 4: Unknown starting position and unknown heading (a cold start)
sigma_x = 100
sigma_y = 100
sigma_a = math.pi
'''


#Constants
sigma_x = 0.1
sigma_y = 0.1
sigma_a = 5*10^(-5)
sigma_s = 32
sigma_omega = 0.55
sigma_r = 10
P0 = np.diag([sigma_x**2, sigma_y**2, sigma_a**2]) # Initial Dynamic covariance matrix
Q = np.diag([sigma_s**2, sigma_omega**2])# (cm^2/s^2,rad^2/s^2)
R = sigma_r**2 # Measurement covariance scalar (cm^2)
J = 1500
T = 0.1 # Sampling rate
Trt = math.sqrt(T) #Root of sampling used for dynamics
Qrt = np.linalg.cholesky(Q)
I = 2 # Number of noise sources

#Load data
M = np.fromfile('xyaswr.bin').reshape((-1,6)).T # Read in truth data
K = M.shape[1]#Number of samples in simulation
x_t = M[0:3] #X truth, Y truth, Heading Truth (cm,cm,rad)
u_t = M[3:5] #Speed truth, Turn rate truth (cm/s,rad/s)
z_t = M[5] #Measurements in cm


#State Initialization
mu = x_t[:,0]
xh_t = np.zeros([3,K]) #State estimate matrix
vh_t = np.zeros([3,K]) #State variance matrix
x_j = np.zeros([3,J])
x_j[0] = mu[0] + np.sqrt(sigma_x)*np.random.randn(J)
x_j[1] = mu[1] + np.sqrt(sigma_y)*np.random.randn(J)
x_j[2] = mu[2] + np.sqrt(sigma_omega)*np.random.randn(J)
Prt = np.linalg.cholesky(P0)
x_j = mu[:, np.newaxis] + Prt @ np.random.randn(3, J)
w = np.ones(J)/J
zh = util.h(x_j)


def resample(w,x_j,J):
    comb = (np.random.rand() + np.arange(J))/J
    je = np.searchsorted(np.cumsum(w), comb)
    x_j = x_j[:, je]
    w = np.ones(J)/J
    return (x_j,w)

for k in range(K):
    # Estimate
    x_rob, y_rob, hdg_rob = x_j[0], x_j[1], x_j[2]

    #Update
    if(k!=K-1):
        res = z_t[k]-util.h(x_j) #Calculate residual
    lh = w*np.exp(-0.5* (res**2)/R) #Calculate the likelihood
    w = lh/sum(lh) #Normalize the set
    xh = x_j @ w #Estimate the new states
    d = xh[:, np.newaxis] - x_j
    Ph = (w*d) @ d.T #estimate of new covariance matrix
    vh_t[:,k] = np.diag(Ph) # Don't know if this is correct

    #Propagate
    nu_j = Trt * Qrt @ np.random.randn(I, J)
    x_j[0, :] = x_rob + (u_t[0, k] + nu_j[0]) * np.cos(hdg_rob) * T
    x_j[1, :] = y_rob + (u_t[0, k] + nu_j[0]) * np.sin(hdg_rob) * T
    x_j[2, :] = hdg_rob + (u_t[1, k] + nu_j[1]) * T

    #Resample
    Je = 1.0/np.sum(w**2)
    if(Je < 0.5 * J):
        x_j,w = resample(w,x_j,J)
    
    # Save results
    xh_t[:, k] = xh
print(xh_t[:,K-1])
util.plot_results(x_t, xh_t, vh_t, z_t, T)
