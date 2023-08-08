import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from numpy import linalg as la
import matplotlib.pyplot as plt

N = 8
h = 1
c = 1
d = 4
Grid = 100
J_z = np.linspace(0,0.1,Grid)
J_p = 0.9
hilbert = 2**N

I = cm.spin("I", 0, N)

sigma_z_1 = cm.spin("Z", c, N)
corr = np.zeros((Grid, d, hilbert))

l=0
for j in J_z:
	print(l)
	H = cm.h_xxz_obc(h, J_p, N, j)
	e, v = la.eigh(H)
	v_dagg = np.conj(v.T)

	for m in range(d):
		sigma_z_2 = cm.spin("Z", c+m+1, N)
		for i in range(hilbert):
			eign_vector = v[:,i]
			eign_vector_dagg = np.conj(eign_vector.T)
			corr[l ,m, i] = cm.check_real(np.dot(eign_vector_dagg, np.dot(sigma_z_1, np.dot(sigma_z_2,eign_vector))))
	l=l+1

with open('/home/abhishek/Documents/ny project/data/new/pert_high_order/N'+str(N)+'_sig_z_corr_s11_d4.pickle', 'wb') as data:
	pickle.dump([corr], data)