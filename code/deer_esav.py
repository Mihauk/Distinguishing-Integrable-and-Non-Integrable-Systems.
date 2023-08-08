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

N = 10
c = 2
d = 3
l = 2
J = 1
t_max =  150
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
S = np.zeros((samples, Ngrid))

h_v = (10*np.random.rand(N)-5)
H = cm.h_rfh_obc( J, N, h_v)
#J_prime = 0.1*J
#H = cm.h_rfxxz( J, J_prime, N, h_v)
e, v = la.eigh(H)
v_dagg = np.conj(v.T)

for j in range(samples):
	print(j)
	s = 1
	r = np.random.randint(2, size=N)
	r[c] = 1
	for ii in range(N):
		if r[ii] == 1:
			s = np.kron(s,cm.up)
		else:
			s = np.kron(s,cm.down)
	psi_0 = s
	phi_0 = np.dot(P_pio2_1,psi_0)
	for i in range(Ngrid):
		phi_tau = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
		chi_tau = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_tau))
		chi_tau_dagg = np.conj(chi_tau.T)
		S[j, i] = cm.check_real(np.dot(chi_tau_dagg, np.dot(sigma_z_1,chi_tau)))

		phi_t = np.dot(P_pi_1,np.dot(P_pi_2,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0)))
		chi_t = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
		chi_t_dagg = np.conj(chi_t.T)
		D[j, i] = cm.check_real(np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))
		#D[j, i] = np.absolute(temp)

sp_eh = S.mean(axis=0)
Deer = D.mean(axis=0)

plt.plot(dt, sp_eh, label=r"$Spin-echo$")
plt.plot(dt, Deer, label=r"$DEER$")
#plt.xticks([50,100,150,200], " ")
#plt.yticks([0.6,0.4,0.2,0,-0.2])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle D(t) \rangle$", fontsize=18)
plt.legend()
plt.show()