import pickle
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x,a,b):
	return a+(b*np.log(x))

#with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav50_ovr_egns_dis2_d_rel_tsi_s13_d5_rc_Ngrid200.pickle', 'rb') as data:
#	temp=pickle.load(data)


#with open('/home/abhiraj654/Documents/data_paper_deer/hv12.5_N8_eav100_ovr_egns_dis100_t1012_d_rel_tsi_s12_d6_spin_echo_Ngrid200.pickle', 'rb') as data:
#	temp=pickle.load(data)


with open('/home/abhiraj654/Documents/data_paper_deer/spin/hv7.5_N8_eav50_ovr_egns_dis500_t108_d_rel_tsi_s12_d5_Ngrid200.pickle', 'rb') as data:
	temp=pickle.load(data)

dt = temp[0]
D = temp[1]
#S = temp[2]
#D2 = temp2[1].reshape(5000, 5, 200)
#D3 = temp3[1].reshape(5000, 4, 200)
#sp = temp[2]
#D1 = temp1[1]
#sp1 = temp1[2]
#s = [2,3,4,5]
'''
dt1 = temp1[0]
D1 = temp1[1]
D2 = temp2[1]
D3 = temp3[1]
D4 = temp4[1]
D5 = temp5[1]
'''

De = D.mean(axis=0)
#Dee = D.mean(axis=1)
Deer = De.mean(axis=0)
#Sp = S.mean(axis=0)
#Spin = Sp.mean(axis=0)
'''
De2 = D2.mean(axis=0, dtype= np.float64)
#Deer2 = De2.mean(axis=0, dtype= np.float64)
De3 = D3.mean(axis=0, dtype= np.float64)
#Deer3 = De3.mean(axis=0, dtype= np.float64)

Deer = np.zeros((4,200))
for i in range(4):
	Deer[i] = (De1[i] + De2[i] + De3[i])/3
'''
'''
spi = sp.mean(axis=0)
spin = spi.mean(axis=0)

De1 = D1.mean(axis=0)
Deer1 = De1.mean(axis=0)

spi1 = sp1.mean(axis=0)
spin1 = spi1.mean(axis=0)
'''

'''
De1 = D1.mean(axis=0)
Deer1 = De1.mean(axis=0)

De2 = D2.mean(axis=0)
Deer2 = De2.mean(axis=0)

De3 = D3.mean(axis=0)
Deer3 = De3.mean(axis=0)

De4 = D4.mean(axis=0)
Deer4 = De4.mean(axis=0)

De5 = D5.mean(axis=0)
Deer5 = De5.mean(axis=0)
'''

#er = stats.sem(De, axis=0)
#er_sp = np.std(Sp, axis=0)
er = np.std(De, axis=0)
'''
er_sp = np.std(spi, axis=0)
er1 = np.std(De1, axis=0)
er_sp1 = np.std(spi1, axis=0)
'''

'''
er1 = np.std(De1, axis=0)
er2 = np.std(De2, axis=0)
er3 = np.std(De3, axis=0)
er4 = np.std(De4, axis=0)
er5 = np.std(De5, axis=0)
'''

'''
i = (np.where(dt>=100)[0])[0]
popt1, pcov1 = curve_fit(func, dt, Deer[1,:])
popt2, pcov2 = curve_fit(func, dt, Deer[2,:])
popt3, pcov3 = curve_fit(func, dt, Deer[3,:])
print(popt1)
print(popt2)
print(popt3)
plt.plot(dt, Deer[1,:], 'ko', label="Original Data, d=1")
plt.plot(dt, func(dt, *popt1), 'r-', label="Fitted Curve, d=1")
plt.plot(dt, Deer[2,:], 'ko', label="Original Data, d=2")
plt.plot(dt, func(dt, *popt2), 'r-', label="Fitted Curve, d=2")
plt.plot(dt, Deer[3,:], 'ko', label="Original Data, d=3")
plt.plot(dt, func(dt, *popt3), 'r-', label="Fitted Curve, d=3")
'''

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


#plt.plot(dt, De[0,2,:], label=r"$r=2$")
#plt.plot(dt, Dee[0,2,:], label=r"$r=2$")
#plt.errorbar(dt, spin, yerr=er_sp, label=r"$spin-echo$")

'''
plt.title(r"$w = 12.5$")
plt.errorbar(dt[0:135], Spin[0:135], yerr=er_sp[0:135], label=r"$spin-echo$")
plt.errorbar(dt[0:135], Deer[1, 0:135], yerr=er[1, 0:135], label=r"$r=2$")
plt.errorbar(dt[0:135], Deer[2, 0:135], yerr=er[2, 0:135], label=r"$r=3$")
plt.errorbar(dt[0:135], Deer[3, 0:135], yerr=er[3, 0:135], label=r"$r=4$")
plt.errorbar(dt[0:135], Deer[4, 0:135], yerr=er[4, 0:135], label=r"$r=5$")
plt.errorbar(dt[0:135], Deer[5, 0:135], yerr=er[5, 0:135], label=r"$r=6$")
#plt.errorbar(dt, Deer[6], yerr=er[6], label=r"$r=7$")
'''
'''
#plt.plot(dt, Deer[0], label=r"$r=1$")
#plt.plot(dt, De[40,2], label=r"$r=1$")
plt.title(r"$w = 7.5$")
#plt.plot(dt[0:135], Spin[0:135], label=r"$spin-echo$")
plt.plot(dt[0:135], Deer[0, 0:135], label=r"$r=1$")
plt.plot(dt[0:135], Deer[1, 0:135], label=r"$r=2$")
plt.plot(dt[0:135], Deer[2, 0:135], label=r"$r=3$")
plt.plot(dt[0:135], Deer[3, 0:135], label=r"$r=4$")
plt.plot(dt[0:135], Deer[4, 0:135], label=r"$r=5$")
#plt.plot(dt[0:135], Deer[5, 0:135], label=r"$r=6$")
#plt.plot(dt, Deer[6], label=r"$r=7$")
'''
#plt.plot(dt, np.zeros((dt.size)))

'''
plt.plot(dt, Deer[1], label=r"$r=2$")
plt.plot(dt, Deer[2], label=r"$r=3$")
plt.plot(dt, Deer[3], label=r"$r=4$")
'''

plt.title(r"$w = 7.5$")
plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$r=1$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$r=2$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$r=3$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$r=4$")
plt.errorbar(dt, Deer[4], yerr=er[4], label=r"$r=5$")
#plt.errorbar(dt, Deer[5], yerr=er[5], label=r"$r=6$")
#plt.errorbar(dt, Deer[6], yerr=er[6], label=r"$r=7$")



'''
plt.errorbar(dt, spin1, yerr=er_sp1, label=r"$spin-echo, pbc$")
plt.errorbar(dt, Deer1[0], yerr=er1[0], label=r"$r=1, pbc$")
plt.errorbar(dt, Deer1[1], yerr=er1[1], label=r"$r=2, pbc$")
plt.errorbar(dt, Deer1[2], yerr=er1[2], label=r"$r=3, pbc$")
plt.errorbar(dt, Deer1[3], yerr=er1[3], label=r"$r=4, pbc$")
plt.errorbar(dt, Deer1[4], yerr=er1[4], label=r"$r=5, pbc$")
'''

'''
sat = np.zeros((4))
for j in range(3):
	sat[j]=dt[(np.where(Deer[j+1,:]<=0)[0])[0]]
sat[3]=10**8.5
print(sat)
plt.plot(s,sat,'o--')
'''

'''
plt.errorbar(dt, Deer[4], yerr=er[4], label=r"$3$")
plt.errorbar(dt, Deer[5], yerr=er[5], label=r"$3.5$")
plt.errorbar(dt, Deer[6], yerr=er[6], label=r"$4$")
plt.errorbar(dt, Deer[7], yerr=er[7], label=r"$5$")
plt.errorbar(dt, Deer[8], yerr=er[8], label=r"$6$")
plt.errorbar(dt, Deer[9], yerr=er[9], label=r"$7$")
plt.errorbar(dt1, Deer1[2], yerr=er1[2], label=r"$10$")
plt.errorbar(dt, Deer[10], yerr=er[10], label=r"$15$")
plt.errorbar(dt, Deer[11], yerr=er[11], label=r"$20$")
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

#plt.ion()
#plt.clf()
#plt.errorbar(s, sat_Deer, yerr=std_sat_Deer, label=r"$hv=5$")
#plt.xticks([0,1,2,3])
#plt.yticks([0,0.2,0.4,0.5])
plt.xlabel(r"$t$", fontsize=18)
#plt.xlabel(r"$r$", fontsize=18)
plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.ylabel(r"$Sat-time$", fontsize=18)
#plt.ylabel(r"$C_{zz}$", fontsize=18)
#plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
