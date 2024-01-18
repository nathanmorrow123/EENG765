#Homework 3 EENG 765
#Nathan Morrow
from scipy.signal import convolve2d as conv
import numpy as np
import util.py

K =10
lsr = [[-3,4],[0,5],[3,4]] #Movement coordinate changes (Left, Straight, Right)
p_lsr = [0.3,0.4,0.3] #Probability movement (Left, Straight, Right)
v = 0.1 # Movement variance m^2
q = 0.1 # Noise variance m^2 
r = 4   # Measurement variance m^2
dx = 0.2 # mesh grid in 0.2 meter squares
mu = 1
x_t, z_t = util.gen_data(K, v, q, r, p_lsr, lsr)
x_limits = util.domain(x_t[0], x_t[1], q, dx)
X, Y = util.mesh(x_limits, dx)
px = util.norm2d(X, Y, mu, v)
pnu_limits = util.domain(range(100),range(100),q,dx)
pnu_l = util.mesh(pnu_limits,dx)
pnu_s = util.mesh(pnu_limits,dx)
pnu_r = util.mesh(pnu_limits,dx)
#h = np.convolve(f, g, mode='same')
px_prior_t = np.zeros((K, X.shape[0], X.shape[1]))
# Step one process results with dynamics only
util.plot_results(X, Y, px_prior_t)

#util.plot_results(X, Y, px_prior_t, px_post_t, z_t)


