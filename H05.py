#Nathan Morrow
#AFIT EENG 765
#Linear Kalman Filter
import numpy as np
import math
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
R = np.matrix([[1,0],[0,0.1]])

#Create State variables
x_t = np.zeros([4,K])
u_t = np.zeros([2,K])
z_t = np.zeros([2,K])

#Set Initial states
x_t[:,0] = np.array([0,0,0.8,0])
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

#Generation
Phi, Bd, Qd = vanloan(F,B,Q,T)
for k in range(K):
    nu = np.diag(Q)*np.random.randn(4)
    eta = np.diag(R)*np.random.randn(2)
    a_x = -math.sin((2*math.pi*k*T)/5)
    a_y = -math.cos((2*math.pi*k*T)/10)
    u = np.array([a_x,a_y])
    x = np.array(x_t[:,k])
    print(Phi)
    exit()
    x_t[k+1] = Phi*x.T+Bd*u.T+nu
    u_t[k] = u+nu
    z_t[k] = H*x+D*u+eta
data = np.vstack((x_t, u_t, z_t))
data.T.tofile("xuz.bin")


#Estimation
data = np.fromfile("xuz.bin").reshape((-1, 8)).T
x_t = data[:4]
u_t = data[4:6]
z_t = data[6:]


