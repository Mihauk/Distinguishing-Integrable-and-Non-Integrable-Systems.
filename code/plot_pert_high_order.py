import pickle
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

with open('/home/abhishek/Documents/ny project/data/new/pert_high_order/N8_sig_z_corr_s11_d4.pickle', 'rb') as data:
	temp=pickle.load(data)

corr = temp[0]

plt.plot([1,2,3,4], corr[10,:,1])

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
'''
plt.show()