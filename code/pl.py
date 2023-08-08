import pickle
import numpy as np
#from scipy import stats
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x,a,b):
	return a+(b*np.log(x))

with open('/home/abhishek/Documents/ny project/data/hv15_N10_eav50_dis1000_t1012_d_rel_tsi_spin_echo_s13_s28_Ngrid200_spa_corr.pickle', 'rb') as data:
	temp=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/hv20_N8_eav50_dis1000_t112_d_rel_tsi_spin_echo_s12_s27_Ngrid200_spa_corr.pickle', 'rb') as data:
	temp1=pickle.load(data)

dt = temp[0]
D = temp[1]
sp = temp[2]
D1 = temp1[1]
sp1 = temp1[2]
#s = np.arange(4)

De = D.mean(axis=0)
Deer = De.mean(axis=0)

spi = sp.mean(axis=0)
spin = spi.mean(axis=0)

De1 = D1.mean(axis=0)
Deer1 = De1.mean(axis=0)

spi1 = sp1.mean(axis=0)
spin1 = spi1.mean(axis=0)

#er = stats.sem(De, axis=0)
er = np.std(De, axis=0)
ers = np.std(spi, axis=0)

er1 = np.std(De, axis=0)
ers1 = np.std(spi, axis=0)

'''
sat_Deer = np.zeros((4))
std_sat_Deer = np.zeros((4))

x = (np.where(dt>=100)[0])[0]
sat_Deer[0] = (Deer[0,x::]).mean()
sat_Deer[1] = (Deer[1,x::]).mean()
sat_Deer[2] = (Deer[2,x::]).mean()
sat_Deer[3] = (Deer[3,x::]).mean()

std_sat_Deer[0] = np.std((Deer[0,x::]))
std_sat_Deer[1] = np.std((Deer[1,x::]))
std_sat_Deer[2] = np.std((Deer[2,x::]))
std_sat_Deer[3] = np.std((Deer[3,x::]))
'''

plt.errorbar(dt, spin, yerr=ers, label=r"$d=4$")
plt.errorbar(dt, Deer, yerr=er, label=r"$DEER,d=4$")

plt.errorbar(dt, spin1, yerr=ers1, label=r"$d=4,N=8$")
plt.errorbar(dt, Deer1, yerr=er1, label=r"$DEER,d=4,N=8$")

#plt.errorbar(s, sat_Deer, yerr=std_sat_Deer, label=r"$hv=5$")
#plt.xticks([0,1,2,3])
#plt.yticks([0,0.2,0.4,0.5])
plt.xlabel(r"$t$", fontsize=18)
#plt.xlabel(r"$d$", fontsize=18)
plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.ylabel(r"$SD$", fontsize=18)
#plt.ylabel(r"$C_{zz}$", fontsize=18)
#plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()