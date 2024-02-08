#Nathan Morrow
#AFIT EENG 765
#Linear Kalman Filter
import numpy as np
import math
import matplotlib.pyplot as plt


#Define Constants
T = 0.1 
t_dur = 100
K = int(t_dur/T) 
theta = math.pi/4
F = np.matrix([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
B = np.matrix([[0,0],[0,0],[1,0],[0,1]])
D = 0
Q = np.diag((np.full(4,0.1)))
H = np.matrix([[math.cos(theta),math.sin(theta),0,0],[-math.sin(theta),math.cos(theta),0,0]])
R = np.diag([1.0, 1.0])


#Load data
data = np.fromfile("xuz.bin").reshape((-1, 8)).T
x_t = data[:4]
u_t = data[4:6]
z_t = data[6:]

#Create Variables
xh_t = np.zeros([4,K])
x_s = np.zeros([4,K])

#Initialize beggining states
xh_t[:,0] = x_t[:,0]
P = np.eye(4)

def vanloan(F, B, Q, T):
    import scipy.linalg as la

    # Phi
    N = F.shape[1] # number of states
    Phi = la.expm(F*T)

    # Bd
    M = B.shape[1] # number of inputs
    G = np.vstack(( np.hstack((F, B)), np.zeros((M, N + M)) ))
    H = la.expm(G*T)
    Bd = H[0:N, N:(N + M)]

    # Qd
    L = np.vstack((
            np.hstack((-F, Q)),
            np.hstack(( np.zeros((N, N)), F.T)) ))
    H = la.expm(L*T)
    Qd = Phi @ H[0:N, N:(2*N)]

    return Phi, Bd, Qd

# Estimation
Phi, Bd, Qd = vanloan(F,B,Q,T)

for k in range(K):
    # Update
    S = H @ P @ H.T + R
    Kf = P @ H.T @ np.linalg.inv(S)
    r = z_t[:, k]- H @ xh_t[:, k]
    xh_t[:, k] = xh_t[:,k] + (Kf @ (r.T)).T
    P -= Kf @ H @ P

    # Propagate
    if k < K-1:
        xh_t[:, k + 1] = Phi @ xh_t[:, k] + Bd @ u_t[:, k]
        P = Phi @ P @ Phi.T + Qd
    
    # Save 3*standard deviation
    x_s[:,k] = 3*np.sqrt(np.diag(P))
    
# Analysis
print(np.round(P,3))  
x_e = x_t - xh_t      
time = T*np.array(range(K))
avg_s = np.zeros(4)
for i in range(len(avg_s)):
    avg_s[i] = (np.sum(x_s[i])/(K))
print(avg_s)

# Plot position truth and estimate
plt.figure()
plt.plot(x_t[0],x_t[1],label='True')
plt.plot(xh_t[0],xh_t[1], label='Estimated')
plt.xlabel("x-axis position (m)")
plt.ylabel("y-axis position (m)")
plt.legend()
plt.savefig("fig_paths.pdf")

# Px error
plt.figure()
plt.plot(time,x_e[0])
plt.fill_between(time,x_s[0],-1*x_s[0],alpha=0.5)
plt.axhline(y=avg_s[0],color='r',zorder = 3)
plt.axhline(y=-1*avg_s[0],color='r',zorder = 3)
plt.xlabel("Time, t (s)")
plt.ylabel("Px error (m)")
plt.savefig("fig_e1.pdf")

# Py error
plt.figure()
plt.plot(time,x_e[1])
plt.fill_between(time,x_s[1],-1*x_s[1],alpha=0.5)
plt.axhline(y=avg_s[1],color='r',zorder = 3)
plt.axhline(y=-1*avg_s[1],color='r',zorder = 3)
plt.xlabel("Time, t (s)")
plt.ylabel("Py error (m)")
plt.savefig("fig_e2.pdf")

# Vx error
plt.figure()
plt.plot(time,x_e[2])
plt.fill_between(time,x_s[2],-1*x_s[2],alpha=0.5)
plt.axhline(y=avg_s[2],color='r',zorder = 3)
plt.axhline(y=-1*avg_s[2],color='r',zorder = 3)
plt.xlabel("Time, t (s)")
plt.ylabel("Vx error (m)")
plt.savefig("fig_e3.pdf")

# Vy error
plt.figure()
plt.plot(time,x_e[3])
plt.fill_between(time,x_s[3],-1*x_s[3],alpha=0.5)
plt.axhline(y=avg_s[3],color='r',zorder = 3)
plt.axhline(y=-1*avg_s[3],color='r',zorder = 3)
plt.xlabel("Time, t (s)")
plt.ylabel("Px error (m)")
plt.savefig("fig_e4.pdf")

