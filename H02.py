import util
import numpy as np
import matplotlib.pyplot as plt
def estimatePos( K, pc, pm, z_t, map):
    pz_x = util.get_pz_x(pc, map) # 5x50 Matrix with a set of colors for each traveling block
    px_t= np.zeros((50,K)) # Pr(x[t]|z)  
    M = util.markov_transition(pm) #50x100 Matrix probability of being at each spot for each point in time
    xh_t = np.zeros(K)
    px = (1/50)*np.ones(50)
    valid = 0
    good = 0
    for k in range(K):
        z = z_t[k] # Color measurement 
        z_x = pz_x[z,:] # Chances of of color throughout the board
        px_t[:,k] = (z_x*px)
        px_t[:,k] /= (px_t[:,k].sum()) # Normalize sums
        px = M@px_t[:,k] # Dynamics model propagates
        xh_t[k] = np.argmax(px_t[:,k])
        if xh_t[k]==x_t[k]:
            valid += 1/px_t[int(xh_t[k]),k]
        if px_t[int(xh_t[k]),k] >= 0.5:
            good += 1
    valid /= K
    good /= K
    return (px_t,xh_t,valid,good)
# Board size is 50 spaces with spots 0-49
K = 100 # Number of moves
pc = 0.8 # Probability of reading the correct color
pm = [0.2,0.3,0.5] # Movement probability [-1,0,1]

# Monte Carlo
N = 100
validness = np.zeros(N)
goodness = np.zeros(N)
for n in range(N):
    x_t, z_t, map = util.gen_truth(K, pc, pm) # x_t truth movement positions across board, #z_t measured colors, #map color at spot
    px_t,xh_t,valid,good  = estimatePos(K,pc,pm,z_t,map)
    validness[n] = valid
    goodness[n] = good
plt.figure()
util.plot_results(px_t,xh_t,x_t)
plt.grid()
plt.savefig("monte_plot1.png")
print("Mone Carlo Trial 1:")
print("Valid:",validness.sum()/N)
print("Good: ",goodness.sum()/N)

# Monte Carlo Part 2
N = 100
validness = np.zeros(N)
goodness = np.zeros(N)
for n in range(N):
    x_t, z_t, map = util.gen_truth(K, pc, pm)
    px_t,xh_t,valid,good  = estimatePos(K,0.6,pm,z_t,map)
    validness[n] = valid
    goodness[n] = good
plt.figure()
util.plot_results(px_t,xh_t,x_t)
plt.grid()
plt.savefig("monte_plot2.png")
print("Monte Carlo Trial 2:")
print("Valid:",validness.sum()/N)
print("Good: ",goodness.sum()/N)


# Monte Carlo Pt 3
N = 100
validness = np.zeros(N)
goodness = np.zeros(N)
for n in range(N):
    x_t, z_t, map = util.gen_truth(K, pc, pm)
    px_t,xh_t,valid,good  = estimatePos(K,0.99,pm,z_t,map)
    validness[n] = valid
    goodness[n] = good
plt.figure()
util.plot_results(px_t,xh_t,x_t)
plt.grid()
plt.savefig("monte_plot3.png")
print("Monte Carlo Trial 3:")
print("Valid:",validness.sum()/N)
print("Good: ",goodness.sum()/N)

# Monte Carlo Pt 4
print("\nMonte Carlo Trial 4:")
N = 100
validness = np.zeros(N)
goodness = np.zeros(N)
PC1 = [.7, .8, .9, .95]
PM1 = [[.3, .4, .3], [.3, .3, .4], [.2, .2, .6], [.1, .1, .8]]
for pc in PC1:
    print("")
    print("PC Value:",pc)
    for pm in PM1:
        print("PM Value:",pm)
        for n in range(N):
            x_t, z_t, map = util.gen_truth(K, pc, pm)
            px_t,xh_t,valid,good  = estimatePos(K,pc,pm,z_t,map)
            validness[n] = valid
            goodness[n] = good
        #print("Valid:",validness.sum()/N)
        print("Good: ",goodness.sum()/N)

