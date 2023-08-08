import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

def pulse(m):
	P = 1/np.sqrt(2)
	P = np.dot(P,(I - 2j*cm.spin("Y", m, N)))
	return (P)

N = 10
c = 4
J = 1
#J_p = 0.1*J
t_max = 6
Ngrid = 200
samples = 50
dis_sam = 100
#dt = np.linspace(0, t_max, Ngrid)
dt = np.logspace(-1, t_max, Ngrid)

I = cm.spin("I", 0, N)


D = np.zeros((dis_sam, samples, 5, Ngrid))
a = np.array([30,50,70,90,100])

for m in range(5):
    print(m)
    n = a[m]
    sigma_z_1 = cm.spin("Z", c, N)
    P_pio2_1 = pulse(c)
    P_pi_1 = np.dot(P_pio2_1, P_pio2_1)
    for k in range(dis_sam):
        print(k)
        h_v = n*(2*np.random.rand(N)-1)
        H = cm.h_rfh_obc( J, N, h_v)
        #H = cm.h_rfxxz_obc( J, J_p, N, h_v)
        e, v = la.eigh(H)
        v_dagg = np.conj(v.T)

        for j in range(samples):
            s = 1
            r = np.random.randint(v[0].size)
            psi_0 = v[:,r]
            psi_0_dagg = np.conj(psi_0.T)
            mz = np.dot(psi_0_dagg, np.dot(2*sigma_z_1, psi_0))
            if (mz<0):
                s = -1
            phi_0 = np.dot(P_pio2_1,psi_0)
            for i in range(Ngrid):
                phi_t = np.dot(P_pi_1, cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_0))
                chi_t = np.dot(P_pio2_1,cm.psi_at_t( e, v, v_dagg, dt[i]/2, phi_t))
                chi_t_dagg = np.conj(chi_t.T)
                D[k, j, m, i] = cm.check_real(s*np.dot(chi_t_dagg, np.dot(sigma_z_1,chi_t)))

with open('/home/abhiraj654/Documents/data_paper_deer/hv30-100_N'+str(N)+'_eav50_ovr_egns_dis100_t100000_s5_rc_Ngrid200_spin_echo.pickle', 'wb') as data:
	pickle.dump([dt, D], data)
