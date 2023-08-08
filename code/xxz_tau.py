import sys
import time
import pickle
import numpy as np
import common as cm
from numpy import linalg as la
from multiprocessing import Pool


def time_evol(e, v, v_dagg, t, psi, psi0_dagg):
	psi_t = cm.psi_at_t(e,v,v_dagg,t,psi)
	psi_t_dagg = np.transpose(np.conjugate(psi_t))
	a = cm.check_real(np.dot(psi_t_dagg, np.dot(m_z_proj, psi_t)))
	b = cm.check_real(np.dot(psi_t_dagg,np.dot(s_c_proj, psi_t)))
	c = cm.bipartite_fl(s_l2_proj, psi_t, psi_t_dagg)
	d = cm.entropy(np.dot(Project.T, psi_t), r)
	e = cm.fidelity(psi0_dagg, np.dot(Project.T, psi_t), N)
	return(a,b,c,d,e)

N = 10
J = 1
#J_prime = x
#h = 1
#dh = 0.1
delta = 1
ddelta = 0.1
tau_max=2
#t_max = 2.5*(10**tau_max)
hilbert = 2**N
samples = 600
seed = 60
NGrid = 300
tau_Grid = 40
assert N%2==0
r = int(2**(N/2))
np.random.seed(31415+seed)
dtau = np.logspace(0,tau_max,tau_Grid)

m_z = cm.magnetization(N)
s_c = cm.spin_corr(N)
s_l2 = cm.sl2(N)

dtf = np.linspace(0,2.5*dtau[tau_Grid-1], NGrid)
window = (np.where(np.logical_and(dtf>=1.75*dtau[39],dtf<=2.25*dtau[39])))[0]

'''
avg_m_z = np.zeros((tau_Grid, samples, NGrid))
S = np.zeros((tau_Grid, samples, NGrid))
b_fl = np.zeros((tau_Grid, samples, NGrid))
avg_s_c = np.zeros((tau_Grid, samples, NGrid))
l = np.zeros((tau_Grid, samples, NGrid))
'''

avg_m_z = np.zeros((tau_Grid, samples, window.size))
S = np.zeros((tau_Grid, samples, window.size))
b_fl = np.zeros((tau_Grid, samples, window.size))
avg_s_c = np.zeros((tau_Grid, samples, window.size))
l = np.zeros((tau_Grid, samples, window.size))
in_energy = np.zeros((samples))
fi_energy = np.zeros((samples))

psi_0 = cm.psi_0_xxz(N)
psi_0_dagg = np.conjugate(np.transpose(psi_0))

'''
Sz_sum = np.zeros((hilbert, hilbert))
for i in range(N):
	sz_i = (np.kron(np.eye(2**i), np.kron(cm.s_z, np.eye(2**(N-i-1)))))
	Sz_sum += sz_i
'''

assert N%2 == 0
ind = np.where( np.abs(N*m_z.diagonal()) < 1e-6 )[0]

dim_proj = ind.size
Project = np.zeros((dim_proj, hilbert))
for j in range(dim_proj): Project[j,ind[j]] = 1.0
psi_0_proj = np.dot(Project, psi_0)
psi_0_proj_dagg = np.conjugate(psi_0_proj.T)

m_z_proj = np.dot(Project, np.dot(m_z, Project.T))
s_c_proj = np.dot(Project, np.dot(s_c, Project.T))
s_l2_proj = np.dot(Project, np.dot(s_l2, Project.T))

for j in range(samples):
	print(j)
	#sys.stdout.flush()
	h_v = 1*(2*np.random.rand(N)-1)
	H = cm.h_rfxxz(J, delta, N, h_v)
	H_T = -cm.h_rfxxz(J, delta+ddelta, N, h_v)

	H_proj = np.dot(Project, np.dot(H, Project.T))
	H_T_proj = np.dot(Project, np.dot(H_T, Project.T))

	e, v = la.eigh(H_proj)
	v_dagg = v.T.conj()

	e_T, v_T = la.eigh(H_T_proj)
	v_dagg_T = v_T.T.conj()

	in_energy[j] = cm.check_real(np.dot(psi_0_proj_dagg,np.dot(H_proj,psi_0_proj)))
	psi_fi = cm.psi_at_t(e_T,v_T,v_dagg_T,dtf[NGrid-1]-dtau[tau_Grid-1],cm.psi_at_t(e,v,v_dagg,dtau[tau_Grid-1],psi_0_proj))
	psi_fi_dagg = np.conjugate(psi_fi.T)
	fi_energy[j] = cm.check_real(np.dot(psi_fi_dagg,np.dot(H_T_proj,psi_fi)))

	for k in range(tau_Grid):
		dt = np.linspace(0, 2.5*dtau[k], NGrid)
		psi_tau_proj = cm.psi_at_t(e,v,v_dagg,dtau[k],psi_0_proj)
		m=0
		for i in window:
			#if dt[i]<=dtau[k]:
			#	avg_m_z[k,j,i], avg_s_c[k,j,i], b_fl[k,j,i], S[k,j,i], l[k,j,i] = time_evol(e,v,v_dagg,dt[i],psi_0_proj,psi_0_dagg)
			#else:
			#	avg_m_z[k,j,i], avg_s_c[k,j,i], b_fl[k,j,i], S[k,j,i], l[k,j,i] = time_evol(e_T,v_T,v_dagg_T,dt[i]-dtau[k],psi_tau_proj,psi_0_dagg)
			avg_m_z[k,j,m], avg_s_c[k,j,m], b_fl[k,j,m], S[k,j,m], l[k,j,m] = time_evol(e_T,v_T,v_dagg_T,dt[i]-dtau[k],psi_tau_proj,psi_0_dagg)
			m = m + 1

with open('/media/abhishek/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/xxz/hv1/xxz_hv1_N'+str(N)+'_seed_'+str(seed)+'_tau_energy_600.pickle', 'wb') as data:
	pickle.dump([dtau, avg_m_z, S, avg_s_c, l, b_fl, in_energy, fi_energy], data)