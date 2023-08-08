import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

N = 10
#c = 1
#d = 4
J = 1
samples = 1500
#dis_sam = 1000

D = np.zeros(( samples, N-1, N-1))

#for k in range(dis_sam):
#	print(k)
h_v = 20*(2*np.random.rand(N)-1)
H = cm.h_rfh_obc( J, N, h_v)
#H = cm.h_rfxxz_obc( J, J_p, N, h_v)
e, v = la.eigh(H)
v_dagg = np.conj(v.T)

for q in range(samples):
	print(q)
	r = np.random.randint(v[0].size)
	psi_0 = v[:,r]
	psi_0_dagg = np.conj(psi_0.T)
	for i in range(N-1):
		sigma_z_1 = cm.spin("Z", i, N)
		for j in range(N-i-1):
			sigma_z_2 = cm.spin("Z", i+j+1, N)
			D[ q, i, j] = cm.check_real(np.dot(psi_0_dagg, np.dot(sigma_z_1, np.dot(sigma_z_2,psi_0))) - (np.dot(psi_0_dagg, np.dot(sigma_z_1, psi_0))*np.dot(psi_0_dagg, np.dot(sigma_z_2, psi_0))))

with open('/home/abhishek/Documents/ny project/data/hv20_N'+str(N)+'_eav5000_ovr_egns_dis1_spa_corr_pal_huse_all.pickle', 'wb') as data:
	pickle.dump([D], data)