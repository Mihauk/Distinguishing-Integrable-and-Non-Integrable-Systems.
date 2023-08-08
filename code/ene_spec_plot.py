import pickle
import numpy as np
from scipy import stats
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt

jp=np.linspace(0,1,40)
temp=[]
x = 0
#with open('/media/abhishek/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/N8/ene/new/annni_hv01_N8_J'+str(jp[x])+'_seed40_tau_energy.pickle', 'rb') as data:
#	temp = (pickle.load(data))

for j in jp:
	with open('/media/abhishek/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/rand_int_st/ene/N8/annni_dh01_ran_inst_N8_J'+str(j)+'_tau_energy.pickle', 'rb') as data:
		temp.append(pickle.load(data))


#N8/ene/new/
#rand_int_st/ene/N8/
#ran_inst_


jpgrid = 40
taugrid = 40
NGrid = 300
dtau = temp[0][0]

dtf = np.linspace(0,2.5*dtau[taugrid-1], NGrid)
window = (np.where(np.logical_and(dtf>=1.75*dtau[taugrid-1],dtf<=2.25*dtau[taugrid-1])))[0]

'''
s = temp[2]
in_ene = temp[6]
fi_ene = temp[7]
tau = 39
sample = in_ene.size

del_e = np.zeros(sample)
s_min = np.zeros(sample)

for i in range(sample):
	del_e[i] = fi_ene[i] + in_ene[i]
	s_min[i] = s[tau, i, :].min()


#plt.clf()
#plt.ion()
plt.plot(del_e, s_min,'.', label=r"$J'=%0.2f$" %jp[x])
plt.xlabel(r"$\delta E$", fontsize=18)
plt.ylabel(r"$S_{min}$", fontsize=18)
plt.legend()
plt.figure()
plt.plot(in_ene,'.', label=r"$J'=%0.2f$" %jp[x])
plt.ylabel(r"$E_i$", fontsize=18)
plt.xlabel(r"$Samples$", fontsize=18)
plt.legend()
plt.figure()
plt.plot(in_ene, del_e, '.', label=r"$J'=%0.2f$" %jp[x])
plt.xlabel(r"$E_i$", fontsize=18)
plt.ylabel(r"$\delta E$", fontsize=18)
plt.legend()
plt.show()
'''


S = np.zeros((jpgrid, taugrid, window.size))
St = np.zeros((jpgrid, taugrid, window.size))
S_min = np.zeros((jpgrid, taugrid))
#S_max = np.zeros((jpgrid, taugrid))
S_min_arg = np.zeros((jpgrid, taugrid), dtype='int')
std = np.zeros((jpgrid, taugrid))

for i in range(jpgrid):
	S[i,:,:] = temp[i][2].mean(axis=1)
	St[i,:,:] = np.std(temp[i][2], axis=1)
	#S_max[i,:] = S[i,:,:].max(axis=-1)
	for j in range(taugrid):
		S_min[i,j] = S[i,j,:].min(axis=-1)
		S_min_arg[i,j] = np.argmin(S[i,j,:], axis=-1)
		std[i,j] = St[i,j,S_min_arg[i,j]]

#plt.plot(dtf[window], S[x,39])
#plt.xlabel(r"$t$", fontsize=18)
#plt.ylabel(r"$S$", fontsize=18)
plt.plot(dtf[window], St[x,39], label=r"$J'=%0.2f$" %jp[x])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\delta S$", fontsize=18)
plt.legend()
plt.figure()
'''
#plt.plot(dtf[window], S[4,39])
#plt.xlabel(r"$t$", fontsize=18)
#plt.ylabel(r"$S$", fontsize=18)
plt.plot(dtf[window], St[4,39], label=r"$J'=%0.2f$" %jp[4])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\delta S$", fontsize=18)
plt.legend()
plt.figure()

#plt.plot(dtf[window], S[10,39])
#plt.xlabel(r"$t$", fontsize=18)
#plt.ylabel(r"$S$", fontsize=18)
plt.plot(dtf[window], St[10,39], label=r"$J'=%0.2f$" %jp[10])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\delta S$", fontsize=18)
plt.legend()
plt.figure()

#plt.plot(dtf[window], S[39,39])
#plt.xlabel(r"$t$", fontsize=18)
#plt.ylabel(r"$S$", fontsize=18)
plt.plot(dtf[window], St[39,39], label=r"$J'=%0.2f$" %jp[39])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\delta S$", fontsize=18)
plt.legend()
'''
plt.show()