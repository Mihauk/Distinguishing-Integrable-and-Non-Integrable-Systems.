import pickle
import numpy as np
from scipy import stats
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt

jp=np.linspace(0,1,40)
with open('/media/abhiraj654/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/xxz/hv10/xxz_hv10_N10_seed_40_tau_energy_600.pickle', 'rb') as data:
	temp = pickle.load(data)

#with open('/media/abhishek/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/xxz/hv1/xxz_hv1_N10_seed_60_tau_energy_600.pickle', 'rb') as data:
#	temp1 = pickle.load(data)


jpgrid = 40
taugrid = 40
NGrid = 300
dtau = temp[0]
tau = 39

dtau = np.logspace(0,2,taugrid)

dtf = np.linspace(0,2.5*dtau[taugrid-1], NGrid)
window = (np.where(np.logical_and(dtf>=1.75*dtau[taugrid-1],dtf<=2.25*dtau[taugrid-1])))[0]



S = np.zeros((taugrid,window.size))
S_min = np.zeros((taugrid))
S_max = np.zeros((taugrid))

S = temp[2].mean(axis=1)
S_max = S.max(axis=-1)
S_min = S[:,window-210].min(axis=-1)

print(S_min)

plt.plot(dtau, 1 - (S_min/S_max))
plt.xlabel(r"$\delta E$", fontsize=18)
plt.ylabel(r"$1 - S_{min}/S_{max}$", fontsize=18)
plt.show()




'''
s = temp[2]
in_ene = temp[6]
fi_ene = temp[7]
sample = in_ene.size

del_e = np.zeros(sample)
s_min = np.zeros(sample)
s_max = np.zeros(sample)

for i in range(sample):
	del_e[i] = fi_ene[i] + in_ene[i]
	s_min[i] = s[tau, i, :].min()
	s_max[i] = s[tau, i, :].max()


#plt.clf()
#plt.ion()
plt.plot(del_e, 1 - (s_min/s_max),'.', label=r"$\tau=%d$" %dtau[tau])
plt.xlabel(r"$\delta E$", fontsize=18)
plt.ylabel(r"$1 - S_{min}/S_{max}$", fontsize=18)
plt.legend()
plt.figure()
plt.plot(in_ene,'.', label=r"$\tau=%d$" %dtau[tau])
plt.ylabel(r"$E_i$", fontsize=18)
plt.xlabel(r"$Samples$", fontsize=18)
plt.legend()
plt.figure()
plt.plot(in_ene, del_e, '.', label=r"$\tau=%d$" %dtau[tau])
plt.xlabel(r"$E_i$", fontsize=18)
plt.ylabel(r"$\delta E$", fontsize=18)
plt.legend()
plt.show()
'''

'''
S = np.zeros((taugrid, window.size))
#S1 = np.zeros((taugrid, window.size))
St = np.zeros((taugrid, window.size))
S_min = np.zeros((taugrid))
S_max = np.zeros((taugrid))

#S_min1 = np.zeros((taugrid))
#S_max1 = np.zeros((taugrid))


S_min_arg = np.zeros((taugrid), dtype='int')
Std = np.zeros((taugrid))

S = temp[2].mean(axis=1)
#S1 = temp1[2].mean(axis=1)
St = np.std(temp[2], axis=1)
S_max = S.max(axis=-1)
S_min = S.min(axis=-1)

#S_max1 = S1.max(axis=-1)
#S_min1 = S1.min(axis=-1)

S_min_arg = np.argmin(S, axis=-1)
for j in range(taugrid):
	Std[j] = St[j,S_min_arg[j]]


plt.plot(dtau, 1 - (S_min/S_max), label=r"$h_v=[-10,10]$")
#plt.plot(dtau, 1 - (S_min1/S_max1), label=r"$h_v=[-1,1]$")
plt.xlabel(r"$\tau$", fontsize=18)
plt.ylabel(r"$1-\frac{S_{min}}{S_{max}}$", fontsize=18)
plt.xscale('log')
plt.figure()
plt.plot(dtf[window], St[tau], label=r"$\tau=%.2f$" %dtau[tau])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\delta S$", fontsize=18)
#plt.figure()
plt.plot(dtf[window], S[tau])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$S$", fontsize=18)
plt.legend()
plt.show()
'''