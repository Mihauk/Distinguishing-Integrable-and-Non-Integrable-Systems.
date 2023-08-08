import pickle
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x,a,b):
	return (a*np.log(x))+b

def fu(r,a):
	return (np.exp(-r/a))

'''
def short_dis(x, y, m, k):
		d = np.abs(y-m*x-k)/np.sqrt(1+m**2)
		return np.argmin(d)
'''

#with open('/home/abhishek/Documents/ny project/data/new/hv10_N10_eav50_dis100_t1012_d_rel_tsi_spin_echo_s11_d9_rc_up_Ngrid200.pickle', 'rb') as data:
#	temp=pickle.load(data)
'''
with open('/home/abhishek/Documents/ny project/data/new/hv10_N10_eav50_dis100_t108_d_rel_tsi_spin_echo_s13_d5_rc_up_Ngrid200.pickle', 'rb') as data:
	temp=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/new/hv10_N10_eav50_dis100_t108_d_rel_tsi_spin_echo_s13_d5_rc_up_Ngrid200_pbc.pickle', 'rb') as data:
	temp=pickle.load(data)
'''
'''
with open('/home/abhiraj654/Documents/data_paper_deer/spin/hv12.5_N8_eav50_ovr_egns_dis500_t108_d_rel_tsi_s12_d5_Ngrid200.pickle', 'rb') as data:
	temp=pickle.load(data)

with open('/home/abhiraj654/Documents/data_paper_deer/spin/hv7.5_N8_eav50_ovr_egns_dis500_t108_d_rel_tsi_s12_d5_Ngrid200.pickle', 'rb') as data:
	temp1=pickle.load(data)
with open('/home/abhiraj654/Documents/data_paper_deer/spin/hv4_N8_eav50_ovr_egns_dis500_t108_d_rel_tsi_s12_d5_Ngrid200.pickle', 'rb') as data:
	temp2=pickle.load(data)
'''
with open('/home/abhiraj654/Documents/data_paper_deer/spin/hv7.5_N8_eav50_dis500_t108_tsi_s12_d6_spin_echo_Ngrid200.pickle', 'rb') as data:
	temp=pickle.load(data)



dt = temp[0]
D1 = temp[1]
#D2 = temp1[1]
#D3 = temp2[1]
sp = temp[2]
'''
D1 = temp1[1]
sp1 = temp1[2]
D2 = temp2[1]
sp2 = temp2[2]
#s = np.arange(3)
'''
#s=np.array([2,3,4,5,6])
#s=np.array([2,3,4,5,6,7,8,9])

spi = sp.mean(axis=0)
spin = spi.mean(axis=0)

De1 = D1.mean(axis=0)
Deer = De1.mean(axis=0)
#De2 = D2.mean(axis=0)
#Deer2 = De2.mean(axis=0)
#De3 = D3.mean(axis=0)
#Deer3 = De3.mean(axis=0)

'''
ar = np.zeros((200,2))
for i in range(200):
	ar[i,0] = dt[i]
	ar[i,1] = Deer[0]

#np.savetxt('w12.5_r4_N8.txt', (dt,Deer[4]))
np.savetxt('w12.5_r1_N8.txt', (dt,Deer[0]))
np.savetxt('w12.5_r2_N8.txt', (dt,Deer[1]))
np.savetxt('w12.5_r3_N8.txt', (dt,Deer[2]))
np.savetxt('w12.5_r4_N8.txt', (dt,Deer[3]))
#np.savetxt('w4_r4_N8.txt', (dt,Deer3[4]))
'''
#spi = sp.mean(axis=0)
#spin = spi.mean(axis=0)
'''
De1 = D1.mean(axis=0)
Deer1 = De1.mean(axis=0)

spi1 = sp1.mean(axis=0)
spin1 = spi1.mean(axis=0)

De2 = D2.mean(axis=0)
Deer2 = De2.mean(axis=0)

spi2 = sp2.mean(axis=0)
spin2 = spi2.mean(axis=0)
'''
#er = stats.sem(De, axis=0)
er = np.std(De1, axis=0)
#er2 = np.std(De2, axis=0)
#er3 = np.std(De3, axis=0)
ers = np.std(spi, axis=0)
'''
er1 = np.std(De1, axis=0)
ers1 = np.std(spi1, axis=0)

er2 = np.std(De2, axis=0)
ers2 = np.std(spi2, axis=0)
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
'''
plt.title(r"$w = 7.5$")
plt.plot(dt[0:135], spin[0:135], label=r"$spin-echo$")
#plt.plot(dt[0:135], Deer[0, 0:135], label=r"$r=1$")
plt.plot(dt[0:135], Deer[1, 0:135], label=r"$r=2$")
plt.plot(dt[0:135], Deer[2, 0:135], label=r"$r=3$")
plt.plot(dt[0:135], Deer[3, 0:135], label=r"$r=4$")
plt.plot(dt[0:135], Deer[4, 0:135], label=r"$r=5$")
plt.plot(dt[0:135], Deer[5, 0:135], label=r"$r=6$")
'''


'''
r=4
plt.title(r"$r = 4$")
#plt.plot(dt, Deer[3], label=r"$w=12.5$")
#plt.plot(dt, Deer2[3], label=r"$w=7.5$")
#plt.plot(dt, Deer3[3], label=r"$w=4$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$w=12.5$")
plt.errorbar(dt, Deer2[3], yerr=er2[3], label=r"$w=7.5$")
plt.errorbar(dt, Deer3[3], yerr=er3[3], label=r"$w=4$")


i = np.where(np.logical_and(dt>=2000,dt<=800000))[0]
popt, pcov = curve_fit(func, dt[i], Deer[3,i])
plt.plot(dt, func(dt, *popt), '-', label=r"fit, $w=12.5$, slope = %f" %(popt[0]))

i1 = np.where(np.logical_and(dt>=150,dt<=25000))[0]
popt1, pcov1 = curve_fit(func, dt[i1], Deer2[3,i1])
plt.plot(dt, func(dt, *popt1), '-', label=r"fit, $w=7.5$, slope = %f" %(popt1[0]))

i2 = np.where(np.logical_and(dt>=40,dt<=1000))[0]
popt2, pcov2 = curve_fit(func, dt[i2], Deer3[3,i2])
plt.plot(dt, func(dt, *popt2), '-', label=r"fit, $w=4$, slope = %f" %(popt2[0]))

plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
plt.xscale('log')
plt.legend()
plt.show()

a = np.zeros((3))
a[2] = popt[0]
a[1] = popt1[0]
a[0] = popt2[0]
plt.plot(np.array([4,7.5,12.5]), a, '.--', label=r"$r = 4$")

plt.xlabel(r"$w$", fontsize=18)
plt.ylabel(r"$Slope$", fontsize=18)
plt.legend()
plt.show()

'''



plt.errorbar(dt, spin, yerr=ers, label=r"$spin-echo$")
#plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$r=1$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$r=2$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$r=3$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$r=4$")
plt.errorbar(dt, Deer[4], yerr=er[4], label=r"$r=5$")
plt.errorbar(dt, Deer[5], yerr=er[5], label=r"$r=6$")
#plt.errorbar(dt, Deer[6], yerr=er[6], label=r"$r=7$")
#plt.errorbar(dt, Deer[7], yerr=er[7], label=r"$r=8$")
#plt.errorbar(dt, Deer[8], yerr=er[8], label=r"$r=9$")
plt.xscale('log')
plt.show()


'''
plt.errorbar(dt, spin2, yerr=ers2, label=r"$spin-echo,d=6$")
plt.errorbar(dt, Deer2, yerr=er2, label=r"$DEER,d=6$")
'''


#i_s = np.where(np.logical_and(dt>=10,dt<=10**12))[0]
#popt_s, pcov_s = curve_fit(func, dt[i_s], spin[i_s])
#plt.plot(dt, func(dt, *popt_s), '-', label=r"fit, spin-echo, slope = %f" %(popt_s[0]))

'''
plt.title(r"$w = 7.5$")
i = np.where(np.logical_and(dt>=15,dt<=250))[0]
popt, pcov = curve_fit(func, dt[i], Deer2[1,i])
plt.plot(dt, func(dt, *popt), '-', label=r"fit, $r=2$, slope = %f" %(popt[0]))

i1 = np.where(np.logical_and(dt>=60,dt<=2000))[0]
popt1, pcov1 = curve_fit(func, dt[i1], Deer2[2,i1])
plt.plot(dt, func(dt, *popt1), '-', label=r"fit, $r=3$, slope = %f" %(popt1[0]))

i2 = np.where(np.logical_and(dt>=150,dt<=25000))[0]
popt2, pcov2 = curve_fit(func, dt[i2], Deer2[3,i2])
plt.plot(dt, func(dt, *popt2), '-', label=r"fit, $r=4$, slope = %f" %(popt2[0]))

i3 = np.where(np.logical_and(dt>=500,dt<=300000))[0]
popt3, pcov3 = curve_fit(func, dt[i3], Deer2[4,i3])
plt.plot(dt, func(dt, *popt3), '-', label=r"fit, $r=5$, slope = %f" %(popt3[0]))

#plt.plot(dt, np.zeros(dt.size))

plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
plt.xscale('log')
plt.legend()
plt.show()


a = np.zeros((4))
a[0] = popt[0]
a[1] = popt1[0]
a[2] = popt2[0]
a[3] = popt3[0]
plt.plot(np.array([2,3,4,5]), a, '.--', label=r"$w = 7.5$")
plt.xlabel(r"$r$", fontsize=18)
plt.ylabel(r"$slope$", fontsize=18)
plt.legend()
plt.show()
'''


#i5 = np.where(np.logical_and(dt>=50000,dt<=4000000000))[0]
#popt5, pcov5 = curve_fit(func, dt[i5], Deer[6,i5])
#plt.plot(dt, func(dt, *popt5), '-', label=r"fit, r=7, slope = %f" %(popt5[0]))

#i6 = np.where(np.logical_and(dt>=250000,dt<=20000000000))[0]
#popt6, pcov6 = curve_fit(func, dt[i6], Deer[7,i6])
#plt.plot(dt, func(dt, *popt6), '-', label=r"fit, r=8, slope = %f" %(popt6[0]))

#i7 = np.where(np.logical_and(dt>=500000,dt<=200000000000))[0]
#popt7, pcov7 = curve_fit(func, dt[i7], Deer[8,i7])
#plt.plot(dt, func(dt, *popt7), '-', label=r"fit, r=9, slope = %f" %(popt7[0]))



'''
J_typ = np.zeros(5)

J_typ[0] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
J_typ[1] = ((popt_s[1]-popt1[1])/popt1[0] + (-popt1[1]/popt1[0]))/2
J_typ[2] = ((popt_s[1]-popt2[1])/popt2[0] + (-popt2[1]/popt2[0]))/2
J_typ[3] = ((popt_s[1]-popt3[1])/popt3[0] + (-popt3[1]/popt3[0]))/2
J_typ[4] = ((popt_s[1]-popt4[1])/popt4[0] + (-popt4[1]/popt4[0]))/2
#J_typ[5] = ((popt_s[1]-popt5[1])/popt5[0] + (-popt5[1]/popt5[0]))/2
#J_typ[6] = ((popt_s[1]-popt6[1])/popt6[0] + (-popt6[1]/popt6[0]))/2
#J_typ[7] = ((popt_s[1]-popt7[1])/popt7[0] + (-popt7[1]/popt7[0]))/2

plt.plot(s, 1/J_typ)

popt_j, pcov_j = curve_fit(fu, s, 1/J_typ)
plt.plot(s, fu(s, *popt_j), '-', label=r"$e^{-r/xi} fit, xi = %f$" %(popt_j))
'''

'''
J_typ[0] = (popt_s[1]-popt[1])/popt[0] - popt[0]*(-popt[1]/popt[0]-(popt_s[1]-popt[1])/popt[0])/2
J_typ[1] = (popt_s[1]-popt1[1])/popt1[0] - popt1[0]*(-popt1[1]/popt1[0]-(popt_s[1]-popt1[1])/popt1[0])/2
J_typ[2] = (popt_s[1]-popt2[1])/popt2[0] - popt2[0]*(-popt2[1]/popt2[0]-(popt_s[1]-popt2[1])/popt2[0])/2
J_typ[3] = (popt_s[1]-popt3[1])/popt3[0] - popt3[0]*(-popt3[1]/popt3[0]-(popt_s[1]-popt3[1])/popt3[0])/2
J_typ[4] = (popt_s[1]-popt4[1])/popt4[0] - popt4[0]*(-popt4[1]/popt4[0]-(popt_s[1]-popt4[1])/popt4[0])/2
J_typ[5] = (popt_s[1]-popt5[1])/popt5[0] - popt5[0]*(-popt5[1]/popt5[0]-(popt_s[1]-popt5[1])/popt5[0])/2
J_typ[6] = (popt_s[1]-popt6[1])/popt6[0] - popt6[0]*(-popt6[1]/popt6[0]-(popt_s[1]-popt6[1])/popt6[0])/2
J_typ[7] = (popt_s[1]-popt7[1])/popt7[0] - popt7[0]*(-popt7[1]/popt7[0]-(popt_s[1]-popt7[1])/popt7[0])/2


J_typ[0] = (-popt[1]/popt[0]) + (popt[0]*(-popt[1]/popt[0]-(popt_s[1]-popt[1])/popt[0]))/2
J_typ[1] = (-popt1[1]/popt1[0]) + (popt1[0]*(-popt1[1]/popt1[0]-(popt_s[1]-popt1[1])/popt1[0]))/2
J_typ[2] = (-popt2[1]/popt2[0]) + (popt2[0]*(-popt2[1]/popt2[0]-(popt_s[1]-popt2[1])/popt2[0]))/2
J_typ[3] = (-popt3[1]/popt3[0]) + (popt3[0]*(-popt3[1]/popt3[0]-(popt_s[1]-popt3[1])/popt3[0]))/2
J_typ[4] = (-popt4[1]/popt4[0]) + (popt4[0]*(-popt4[1]/popt4[0]-(popt_s[1]-popt4[1])/popt4[0]))/2
J_typ[5] = (-popt5[1]/popt5[0]) + (popt5[0]*(-popt5[1]/popt5[0]-(popt_s[1]-popt5[1])/popt5[0]))/2
J_typ[6] = (-popt6[1]/popt6[0]) + (popt6[0]*(-popt6[1]/popt6[0]-(popt_s[1]-popt6[1])/popt6[0]))/2
J_typ[7] = (-popt7[1]/popt7[0]) + (popt7[0]*(-popt7[1]/popt7[0]-(popt_s[1]-popt7[1])/popt7[0]))/2
'''



'''
slp = np.zeros((8))
slp1 = np.zeros((8))
slp[0]=popt[0]
slp[1]=popt1[0]
slp[2]=popt2[0]
slp[3]=popt3[0]
slp[4]=popt4[0]
slp[5]=popt5[0]
slp[6]=popt6[0]	
slp[7]=popt7[0]

slp1[0]=-0.103027
slp1[1]=-0.059047
slp1[2]=-0.051677
slp1[3]=-0.037904

slp1[4]=-0.096183
slp1[5]=-0.063211
slp1[6]=-0.051134
slp1[7]=-0.049996


a = np.zeros((8))
for i in range(8):
	a[i]=1/np.sqrt(i+2)

print(a)

plt.plot(a[0:4],slp1[0:4],'o--', label=r"$FBC, bulk-spin$")
plt.plot(a[0:4],slp1[4:8],'o--', label=r"$PBC, bulk-spin$")
plt.plot(a, slp[0:8],'o--', label=r"$FBC, edge-spin$")
'''

'''
intrcpt = np.zeros((4))
intrcpt[0]=541.363#678.616 #5.02781 #bulk #pbc#541.363 #8.692 #obc#376.914 #8.33858
intrcpt[1]=15995#14960.2 #20.7969 #bulk #pbc15995 #34.691 #obc#18007.9 #21.9389
intrcpt[2]=171585#729864 #127.42 #bulk #pbc171585 #101.303 #obc#587916 #337.7
intrcpt[3]=497866#15273700 #265.612 #bulk #pbc497866 #234.224 #obc#12416800 #523.132
'''
'''
intrcpt[4]=#279224000 #1302.76
intrcpt[5]=#7337880000 #13055.2
intrcpt[6]=#43986500000 #12071.8
intrcpt[7]=#599587000000 #70171.4
'''
#plt.plot(s,intrcpt,'o--')



#plt.errorbar(s, sat_Deer, yerr=std_sat_Deer, label=r"$hv=5$")
#plt.xticks([0,1,2,3])
#plt.yticks([0,0.2,0.4,0.5])
#plt.yticks([10**(-1),10**(-0.5),10**0])
#plt.xlabel(r"$t$", fontsize=18)
#plt.xlabel(r"$\frac{1}{\sqrt{r}}$", fontsize=18)
plt.xlabel(r"$r$", fontsize=18)
plt.ylabel(r"$J_{typ}$", fontsize=18)
#plt.ylabel(r"$Slope$", fontsize=18)
#plt.ylabel(r"$Sat-time$", fontsize=18)
#plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.ylabel(r"$SD$", fontsize=18)
#plt.ylabel(r"$C_{zz}$", fontsize=18)
#plt.yscale('log', basey=np.exp(1))
#plt.yscale('log')
plt.xscale('log')
#plt.legend()
#plt.show()