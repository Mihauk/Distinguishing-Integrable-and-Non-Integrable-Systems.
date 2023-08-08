import pickle
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_dis100_t100000_d_rel_tsi_s14_rc_Ngrid200_spa_corr.pickle', 'rb') as data:
	temp=pickle.load(data)

dt = temp[0]
D = temp[1]
#s = np.arange(4)

De = D.mean(axis=0)
Deer = De.mean(axis=0)


#er = stats.sem(De, axis=0)
er = np.std(De, axis=0)

#X,Y = np.meshgrid(dt, d)
#fig, ax = plt.subplots()

#p = ax.pcolor(X, Y, Deer)
#cb = fig.colorbar(p)

#plt.plot(d, Deer[:,199], label=r"$t=150$")
'''
plt.plot(dt, er1[3], label=r"$spin-echo$")
plt.plot(dt, er[0], label=r"$d=0$")
plt.plot(dt, er[1], label=r"$d=1$")
plt.plot(dt, er[2], label=r"$d=2$")
plt.plot(dt, er[3], label=r"$d=3$")
'''
#plt.errorbar(dt, Deer, yerr=er, label=r"$hv=5$")
plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$d=0$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$d=1$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$d=2$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$d=3$")


#plt.errorbar(s, sat_Deer, yerr=std_sat_Deer, label=r"$hv=5$")
#plt.xticks([0,1,2,3])
#plt.yticks([0,0.2,0.4,0.5])
plt.xlabel(r"$t$", fontsize=18)
#plt.xlabel(r"$d$", fontsize=18)
plt.ylabel(r"$\langle\langle s_i^{z} s_j^{z} \rangle\rangle$", fontsize=18)
#plt.ylabel(r"$SD$", fontsize=18)
#plt.ylabel(r"$C_{zz}$", fontsize=18)
plt.xscale('log')
plt.legend()
plt.show()