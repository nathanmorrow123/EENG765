#Homework 3 EENG 765
#Nathan Morrow
from scipy.signal import convolve2d as conv
import numpy as np
import util
import matplotlib.pyplot as plt
K =10
lsr = [[-3,4],[0,5],[3,4]] #Movement coordinate changes (Left, Straight, Right)
p_lsr = [0.3,0.4,0.3] #Probability movement (Left, Straight, Right)
v = 0.1 # Movement variance m^2
q = 0.1 # Noise variance m^2 
r = 4   # Measurement variance m^2
dx = 0.2 # mesh grid in 0.2 meter squares
mu = [0,0] #Not certain value for mu
x_t, z_t = util.gen_data(K, v, q, r, p_lsr, lsr)
x_limits = util.domain(x_t[0], x_t[1], q, dx)
z_limits = util.domain(z_t[0], z_t[1], q, dx)
X, Y = util.mesh([-20,20,-2,50], dx)
px = util.norm2d(X, Y, mu, v)
pnu_X, pnu_Y = util.mesh([-5,5,-10,10],dx)
pnu_l = 0.3*util.norm2d(pnu_X,pnu_Y,[-3,4],v)
pnu_s = 0.4*util.norm2d(pnu_X,pnu_Y,[0,5],v)
pnu_r = 0.3*util.norm2d(pnu_X,pnu_Y,[3,4],v)
pnu = pnu_l+pnu_s+pnu_r
px_prior_t = np.zeros((K, X.shape[0], X.shape[1]))
px_post_t = np.zeros((K,X.shape[0],X.shape[1]))
px_prior_t[0] = px
for k in range(K):
    #Update
    z = util.norm2d(X,Y,z_t[:,k],r)
    px_prior = conv(px,pnu, mode = 'same')
    #Store
    print(k)
    px = px_prior
    if(k+1<K):
        px_prior_t[k+1] = px_prior
# Step one process results with dynamics only
util.plot_results(X, Y, px_prior_t)
plt.savefig("Figure_1.png")
# Step two process results with measurement dynamics
px = util.norm2d(X, Y, mu, v)
px_prior_t[0] = px
z = util.norm2d(X,Y,z_t[:,0],r)
px_post_t[0] = z*px
for k in range(K):
    #Update
    px_prior = px
    z = util.norm2d(X,Y,z_t[:,k],r)
    px = z*px_prior
    px /= np.linalg.norm(px)
    px_post = px
    px = conv(px,pnu, mode = 'same')
    px /= np.linalg.norm(px)
    #Store
    print(k)
    px_prior_t[k] = px_prior
    px_post_t[k] = 2*px_post

util.plot_results(X, Y, px_prior_t, px_post_t, z_t)
plt.savefig("Figure_2.png")
plt.show()


