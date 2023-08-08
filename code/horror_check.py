import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

def pulse(i, m):
	P = 1/np.sqrt(2)
	if i == 1:
		P = np.dot(P,(I - 2j*cm.spin("Y", c, N)))
	elif i == 2:
		#P = P**l
		#for ii in range(l):
		P = np.dot(P,(I - 2j*cm.spin("Y", (1+c+m), N))) #
	return (P)

N = 10
c = 1
d = 7
#l = 1
J = 1
#J_p = 0.1*J
#t_max = 5
Ngrid = 200
samples = 4
#dt = np.linspace(0, t_max, Ngrid)
dt = np.logspace(-1, 16, Ngrid)

I = cm.spin("I", 0, N)

sigma_z_1 = cm.spin("Z", c, N)
D = np.zeros((samples, d, Ngrid))
sp = np.zeros((samples, Ngrid))


h_v = np.array([-19.74866623332742, -6.461050421280916, 23.90603868479093, 14.44622271439229, 22.01890864732869, 13.87624672997578, -20.82876696504763, -4.27580739089494, -11.11451726860179, -17.48151183518299])/2

H = cm.h_rfh_obc( J, N, h_v)
#H = cm.h_rfxxz_obc( J, J_p, N, h_v)
e, v = la.eigh(H)
v_dagg = np.conj(v.T)

sel = np.zeros((4), dtype='int')
sel[0] = np.where(e*4 == 40.973947771801065)[0]
sel[1] = np.where(e*4 == 53.818388451969987)[0]
sel[2] = np.where(e*4 == 59.219003455046106)[0]
sel[3] = np.where(e*4 == 72.063444135215192)[0]
print(sel)

for m in range(d):
	print(m)
	#sys.stdout.flush()
	#sigma_z_2 = cm.spin("Z", c+m+1, N)
	P_pio2_1 = pulse(1,c)
	P_pio2_2 = pulse(2,m)
	P_pi_1 = np.dot(P_pio2_1, P_pio2_1)
	P_pi_2 = np.dot(P_pio2_2, P_pio2_2)

	for j in range(samples):
		print(j)
		s = 1
		#r = np.random.randint(v[0].size)
		#print(r)
		'''
		s_pro = 1
		r = np.random.randint(2, size=N)
		for ii in range(N):
			if r[ii] == 1:
				s_pro = np.kron(s_pro,cm.up)
			else:
				s_pro = np.kron(s_pro,cm.down)
			psi_0 = s_pro
		'''
		psi_0 = v[:,sel[j]]
		psi_0_dagg = np.conj(psi_0.T)
		mz = np.dot(psi_0_dagg, np.dot(2*sigma_z_1, psi_0))
		if (mz<0):
			s = -1

		phi_0 = np.dot(P_pio2_1,psi_0)
		for i in range(Ngrid):
			if(m==0):
				phi_t_s = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
				chi_t_s = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t_s))
				chi_t_s_dagg = np.conj(chi_t_s.T)
				sp[j, i] = cm.check_real(s*np.dot(chi_t_s_dagg, np.dot(sigma_z_1,chi_t_s)))

			phi_t = np.dot(P_pi_1,np.dot(P_pi_2,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0)))
			chi_t = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
			chi_t_dagg = np.conj(chi_t.T)
			D[j, m, i] = cm.check_real(s*np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))

with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N'+str(N)+'_eav4_ovr_egns_spin_echo_weight_dis1_d_rel_tsi_s12_d6_Ngrid200_horr_chk.pickle', 'wb') as data:
	pickle.dump([dt, D, sp], data)