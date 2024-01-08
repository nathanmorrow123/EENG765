#Nathan Morrow 
#AFIT EENG 765
#Dr. David Woodburn
import numpy as np
import matplotlib.pyplot as plt
def store(M, filename):
    M.tofile(filename + ".bin")
# constants
T = 1e-3    # sampling period (s)
t_dur = 0.2 # duration of simulation (s)
R = 1.0     # resistance (ohm)
L = 10e-3   # inductance (H)
C = 10e-3   # capacitance (F)
# time array
K = round(t_dur/T) + 1  # points in time
t = np.arange(K)*T
# states and inputs
x = np.zeros(2) # iL (A), vC (V)
vs = 0      # source voltage (V)
# storage
x_t = np.zeros((2, K))
vs_t = np.zeros(K)
for k in range(K):
    # updates
    iR = x[1]/R
    if k % 50 == 0: # every 50 ms
        vs = 1 - vs
    vs_t[k] = vs
    # storage
    x_t[:, k] = x
    # derivatives
    Dx = np.array([(vs - x[1])/L, (x[0] - iR)/C])
    # integrals
    x += Dx*T
# Save to file.
store(x_t, "states")
# Plot source voltage switching every 50ms
plt.figure()
plt.plot(t*1000, vs_t)
plt.xlabel('Time, $t$ (ms)')
plt.ylabel('Source Voltage (V)')
plt.savefig('Source_Voltages.png')
# Plot two states for current through inductor and cap voltage
plt.figure()
plt.plot(t*1000, x_t.T)
plt.xlabel('Time, $t$ (ms)')
plt.ylabel('State values')
plt.legend(('Current Through Inductor(A)', 'Capicitor Voltage (V)'))
plt.savefig('States.png')
# Plot resitor and capicitor currents together over time
iR_t = np.zeros(K)
iC_t = np.zeros(K)
for k in range(K):
    iL = x_t[0,k]
    vC = x_t[1,k]
    iR_t[k] = vC/R
    iC_t[k] = iL-iR_t[k]
plt.figure()
plt.plot(t*1000,iR_t,iC_t)
plt.xlabel('Time, $t$ (ms)')
plt.ylabel('Current (A)')
plt.legend(('Resistor Current', 'Capacitor Current'))
plt.savefig('Currents.png')
# Plot the impedance of the resistor in parallel with the capacitor over time,
zR_t = np.zeros(K)
for k in range(K):
    iL = x_t[0,k]
    vC = x_t[1,k]
    zR_t[k]= vC/iL
plt.figure()
plt.plot(t*1000, zR_t)
plt.xlabel('Time, $t$ (ms)')
plt.ylabel('Impedance (Ohms)')
plt.savefig('Impedance.png')
plt.show()
