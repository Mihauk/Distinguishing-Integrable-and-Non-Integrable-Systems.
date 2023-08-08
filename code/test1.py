import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

def pulse(i):
	P = 1/np.sqrt(2)
	if i == 1:
		P = np.dot(P,(I - 2j*cm.spin("Y", c, N)))
	elif i == 2:
		P = P**l
		for ii in range(l):
			P = np.dot(P,(I - 2j*cm.spin("Y", (ii+1+c+d)%N, N)))
	return (P)

def time_evol( t, p_0):
	psi_t = cm.psi_at_t( e, v, v_dagg, t, p_0)
	psi_t_dagg = np.conj(psi_t.T) #transpose not neccesary here as each realization is nx1 matrix
	a = cm.check_real(np.dot(psi_t_dagg, np.dot(sigma_z_1,psi_t)))
	return (a)

N = 10
c = 2
d = 3
l = 1
J = 1
t1 = 10
tau = 60

t_max = 2.5*tau
Ngrid = 300
samples = 50
dt = np.linspace(0, t_max, Ngrid)

I = cm.spin("I", 0, N)

P_pio2_1 = pulse(1)
P_pio2_2 = pulse(2)
P_pi_1 = np.dot(P_pio2_1, P_pio2_1)
P_pi_2 = np.dot(P_pio2_2, P_pio2_2)


sigma_z_1 = cm.spin("Z", c, N)
D = np.zeros((samples, Ngrid))

h_v = (10*np.random.rand(N)-5)
#psi_0 = np.zeros((2**N))
#psi_0[0] = 1
#psi_0 = cm.psi_0_xxz(N)
#psi_0 = cm.rnd_egnst(samples, N)
#psi_0_dagg = np.conj(psi_0) #transpose not neccesary here as nx1 matrix

H = cm.h_rfh( J, N, h_v)
#J_prime = 0.1*J
#H = cm.h_rfxxz( J, J_prime, N, h_v)
e, v = la.eigh(H)
v_dagg = np.conj(v.T)

m = (np.where(dt <= tau)[0])[-1]
m_p = (np.where(dt <= (2*tau-t1))[0])[-1]

for j in range(samples):
	s = 1
	r = np.random.randint(2, size=N)
	for ii in range(N):
		if r[ii] == 1:
			s = np.kron(s,cm.up)
		else:
			s = np.kron(s,cm.down)
	psi_0 = s

	phi_0 = np.dot(P_pio2_1,psi_0)
	psi_tau = cm.psi_at_t( e, v, v_dagg, dt[m], phi_0)
	phi_t = np.dot(P_pi_1,np.dot(P_pi_2,psi_tau))
	psi_tau_p = cm.psi_at_t( e, v, v_dagg, dt[m_p], phi_t)
	chi_t = np.dot(P_pio2_1,psi_tau_p)
	for i in range(Ngrid):
		if dt[i] <= t1:
			D[j, i] = time_evol( dt[i], psi_0)
		elif dt[i] <= tau:
			D[j, i] = time_evol( dt[i], phi_0)
		elif dt[i] <= (2*tau-t1):
			D[j, i] = time_evol( dt[i], phi_t)
		else:
			D[j, i] = time_evol( dt[i], chi_t)


Deer = D.mean(axis=0)

plt.plot(dt, Deer)
#plt.xticks([50,100,150,200], " ")
#plt.yticks([0.6,0.4,0.2,0,-0.2])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle D(t) \rangle$", fontsize=18)
#plt.legend()
plt.show()