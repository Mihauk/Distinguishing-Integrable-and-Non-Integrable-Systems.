import pickle
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x,a,b):
	return a+(b*np.log(x))

with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_dis100_t100000_d_rel_tsi_s14_rc_Ngrid200.pickle', 'rb') as data:
	temp=pickle.load(data)

'''
with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_dis100_t100000_d_rel_tsi_s11_rc_Ngrid200.pickle', 'rb') as data:
	temp1=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_dis100_t100000_s1to4_rc_Ngrid200_spin_echo.pickle', 'rb') as data:
	temp2=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_ovr_egns_dis100_t100000_s1to4_rc_Ngrid200_spin_echo.pickle', 'rb') as data:
	temp3=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_ovr_egns_dis100_t100000_d_rel_tsi_s14_rc_Ngrid200.pickle', 'rb') as data:
	temp4=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_ovr_egns_dis100_t100000_d_rel_tsi_s11_rc_Ngrid200.pickle', 'rb') as data:
	temp5=pickle.load(data)
'''

dt = temp[0]
D = temp[1]
'''
D1 = temp1[1]
#s = np.arange(4)
'''

De = D.mean(axis=0)
Deer = De.mean(axis=0)

'''
De1 = D1.mean(axis=0)
Deer1 = De1.mean(axis=0)
'''
#er = stats.sem(De, axis=0)
er = np.std(De, axis=0)
'''
er1 = np.std(De1, axis=0)
'''

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
#plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$d=1$")
#plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$d=2$")
#plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$d=3$")


'''
plt.errorbar(dt, Deer2[3], yerr=er2[3], label=r"$bulk, spin-echo$")
plt.errorbar(dt, Deer2[0], yerr=er2[0], label=r"$edge, spin-echo$")

plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$bulk, Deer, d=1$")
plt.errorbar(dt, Deer1[1], yerr=er1[1], label=r"$edge, Deer, d=1$")


plt.errorbar(dt, Deer3[3], yerr=er3[3], label=r"$bulk, spin-echo, over-egnst$")
plt.errorbar(dt, Deer3[0], yerr=er3[0], label=r"$edge, spin-echo, over-egnst$")

plt.errorbar(dt, Deer4[1], yerr=er4[1], label=r"$bulk, Deer, d=1, over-egnst$")
plt.errorbar(dt, Deer5[1], yerr=er5[1], label=r"$edge, Deer, d=1, over-egnst$")
'''


'''
plt.errorbar(dt, Deer3[0], yerr=er3[0], label=r"$spin-echo$")
plt.errorbar(dt, Deer5[0], yerr=er5[0], label=r"$d=0$")
plt.errorbar(dt, Deer5[1], yerr=er5[1], label=r"$d=1$")
plt.errorbar(dt, Deer5[2], yerr=er5[2], label=r"$d=2$")
plt.errorbar(dt, Deer5[3], yerr=er5[3], label=r"$d=3$")
plt.errorbar(dt, Deer5[4], yerr=er5[4], label=r"$d=4$")
plt.errorbar(dt, Deer5[5], yerr=er5[5], label=r"$d=5$")
plt.errorbar(dt, Deer5[6], yerr=er5[6], label=r"$d=6$")
'''

'''
plt.errorbar(dt, De[50,0], yerr=er[0], label=r"$d=0$")
plt.errorbar(dt, De[50,1], yerr=er[1], label=r"$d=1$")
plt.errorbar(dt, De[50,2], yerr=er[2], label=r"$d=2$")
plt.errorbar(dt, De[50,3], yerr=er[3], label=r"$d=3$")
'''

'''
plt.errorbar(dt, De[40,0], label=r"$d=0$")
plt.errorbar(dt, De[40,1], label=r"$d=1$")
plt.errorbar(dt, De[40,2], label=r"$d=2$")
plt.errorbar(dt, De[40,3], label=r"$d=3$")
'''

'''
fig, ax1 = plt.subplots()
ax1.plot(dt, Deer[2], 'b-')
ax1.set_xlabel('t(d=2)',  color='b', fontsize=12)
ax1.tick_params('x', colors='b')
plt.xscale('log')

ax2 = ax1.twiny()
ax2.plot(dt, Deer[3], 'r.')
ax2.set_xlabel('t(d=3)', color='r', fontsize=12)
ax2.tick_params('x', colors='r')
'''
i = np.where(np.logical_and(dt>30,dt<20000))[0]
j = np.where(np.logical_and(dt>4,dt<2500))[0]
k = np.where(np.logical_and(dt>80,dt<50000))[0]

dt1 = np.exp(-2)*dt
dt2 = np.exp(-4)*dt
dt3 = np.exp(-5)*dt
dt4 = np.exp(-5)*dt
dt5 = np.exp(-6)*dt
dt6 = np.exp(-7)*dt

#plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$d=0$")
plt.errorbar(dt1, Deer[1], yerr=er[1], label=r"$d=1$")
plt.errorbar(dt2, Deer[2], yerr=er[2], label=r"$d=2$")
plt.errorbar(dt3, Deer[3], yerr=er[3], label=r"$d=3$")
#plt.errorbar(dt4, Deer[4], yerr=er[4], label=r"$d=4$")
#plt.errorbar(dt5, Deer[5], yerr=er[5], label=r"$d=5$")
#plt.errorbar(dt6, Deer[6], yerr=er[6], label=r"$d=6$")



#plt.errorbar(s, sat_Deer, yerr=std_sat_Deer, label=r"$hv=5$")
plt.xticks(dt)
#plt.yticks([0,0.2,0.4,0.5])
plt.xlabel(r"$t$", fontsize=18)
#plt.xlabel(r"$d$", fontsize=18)
plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.ylabel(r"$SD$", fontsize=18)
#plt.ylabel(r"$C_{zz}$", fontsize=18)
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.show()