import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

N = 8
c = 1
d = 5
J = 1
samples = 50
dis_sam = 1000

sigma_z_1 = cm.spin("+", c, N)

D = np.zeros((dis_sam, samples, d))

for k in range(dis_sam):
	print(k)
	h_v = 10*(2*np.random.rand(N)-1)
	H = cm.h_rfh_obc( J, N, h_v)
	#H = cm.h_rfxxz_obc( J, J_p, N, h_v)
	e, v = la.eigh(H)
	v_dagg = np.conj(v.T)

	for j in range(samples):
		r = np.random.randint(v[0].size)
		psi_0 = v[:,r]
		psi_0_dagg = np.conj(psi_0.T)

		for i in range(d):
			sigma_z_2 = cm.spin("-", c+i+1, N)
			D[k, j, i] = cm.check_real(np.dot(psi_0_dagg, np.dot(sigma_z_1, np.dot(sigma_z_2,psi_0))) - (np.dot(psi_0_dagg, np.dot(sigma_z_1, psi_0))*np.dot(psi_0_dagg, np.dot(sigma_z_2, psi_0))))

with open('/home/abhishek/Documents/ny project/data/sig_plus_min_corr/hv10_N'+str(N)+'_eav50_ovr_egns_dis1000_sig_plus_min_corr_s12_pal_huse.pickle', 'wb') as data:
	pickle.dump([D], data)