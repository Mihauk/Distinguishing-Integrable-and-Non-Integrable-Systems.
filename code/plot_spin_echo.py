import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#with open('/home/abhiraj654/Documents/data_paper_deer/hv0-5-10-5_N8_eav50_ovr_egns_dis100_t100000_s4_rc_Ngrid200_spin_echo.pickle', 'rb') as data:
#	temp=pickle.load(data)
#with open('/home/abhiraj654/Documents/data_paper_deer/hv20_N8_eav50_ovr_egns_dis100_t100000_s4_rc_Ngrid200_spin_echo.pickle', 'rb') as data:
#	temp1=pickle.load(data)

with open('/home/abhiraj654/Documents/data_paper_deer/hv30-100_N10_eav50_ovr_egns_dis100_t100000_s5_rc_Ngrid200_spin_echo.pickle', 'rb') as data:
	temp=pickle.load(data)



dt = temp[0]
D = temp[1]
#dt1 = temp1[0]
#D1 = temp1[1]

#s = np.arange(4)

De = D.mean(axis=0)
Deer = De.mean(axis=0)
#De1 = D1.mean(axis=0)
#Deer1 = De1.mean(axis=0)

#e = stats.sem(D, axis=0)
er = np.std(De, axis=0)
#er1 = np.std(De1, axis=0)

#sat_Deer = np.zeros((4))
#std_sat_Deer = np.zeros((4))
#sat_Deer1 = np.zeros((4))
#std_sat_Deer1 = np.zeros((4))
'''
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

#y = (np.where(dt1>=100)[0])[0]
#sat_Deer1[0] = (Deer1[0,y::]).mean()
#sat_Deer1[1] = (Deer1[1,y::]).mean()
#sat_Deer1[2] = (Deer1[2,y::]).mean()
#sat_Deer1[3] = (Deer1[3,y::]).mean()

#std_sat_Deer1[0] = np.std((Deer1[0,x::]))
#std_sat_Deer1[1] = np.std((Deer1[1,x::]))
#std_sat_Deer1[2] = np.std((Deer1[2,x::]))
#std_sat_Deer1[3] = np.std((Deer1[3,x::]))


#X,Y = np.meshgrid(dt, d)
#fig, ax = plt.subplots()

#p = ax.pcolor(X, Y, Deer)
#cb = fig.colorbar(p)

#plt.plot(d, Deer[:,199], label=r"$t=150$")
'''
plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$30$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$50$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$70$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$90$")
plt.errorbar(dt, Deer[4], yerr=er[4], label=r"$100$")
'''
'''
plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$h_v=0.5$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$h_v=1.5$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$h_v=2.5$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$h_v=3.5$")
plt.errorbar(dt, Deer[4], yerr=er[4], label=r"$h_v=4.5$")
plt.errorbar(dt, Deer[5], yerr=er[5], label=r"$h_v=5.5$")
plt.errorbar(dt, Deer[6], yerr=er[6], label=r"$h_v=6.5$")
plt.errorbar(dt, Deer[7], yerr=er[7], label=r"$h_v=7.5$")
plt.errorbar(dt, Deer[8], yerr=er[8], label=r"$h_v=8.5$")
plt.errorbar(dt, Deer[9], yerr=er[9], label=r"$h_v=9.5$")
plt.errorbar(dt, Deer1, yerr=er1, label=r"$h_v=20$")
'''

h = np.array(([30,50,70,90,100]))
ih = 1/h
sp = np.zeros((5))
for i in range(5):
    sp[i] = np.mean(Deer[i,100:199])
#sp[9] = np.mean(Deer1[100:199])

plt.plot(ih, sp)

#plt.errorbar(s, sat_Deer, yerr=std_sat_Deer, label=r"$hv=5$")
#plt.errorbar(s, sat_Deer1, yerr=std_sat_Deer1, label=r"$hv=10$")
#plt.xticks([0,1,2,3])
#plt.yticks([0.2,0.4,0.5])
#y=D.mean(axis=1)
#plt.plot(dt, D[22,3,3])
plt.xlabel(r"$1/h$", fontsize=18)
#plt.xlabel(r"$s$", fontsize=18)
plt.ylabel(r"$spin-echo_{sat}$", fontsize=18)
#plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.xscale('log')
#plt.legend()
#plt.savefig("echo_disorder.pdf")
plt.show()
