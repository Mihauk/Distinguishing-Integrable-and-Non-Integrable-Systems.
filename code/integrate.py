import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.optimize import curve_fit

def func(x,a,b):
	return (a*np.log(x))+b

def fu(r,a):
	return (np.exp(-r/a))

#with open('/home/abhishek/Documents/ny project/data/new/test/hv10_N10_eav50_dis100_t108_1016_d_rel_tsi_spin_echo_s11_d8_rc_up_Ngrid200_underflow.pickle', 'rb') as data:
#	temp=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav50_dis2_d_rel_tsi_s11_d9_rc_Ngrid200.pickle', 'rb') as data:
#	temp=pickle.load(data)


#with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav50_dis11_d_rel_tsi_s13_d5_rc_Ngrid200.pickle', 'rb') as data:
#	temp=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/new/hv10_N10_eav50_dis100_t108_d_rel_tsi_spin_echo_s13_d5_rc_up_Ngrid200_pbc.pickle', 'rb') as data:
#	temp = pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav4_ovr_egns_dis1_d_rel_tsi_s12_s29_rc_Ngrid200.pickle', 'rb') as data:
#	temp=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav4_ovr_egns_weight_dis1_d_rel_tsi_s12_s29_rc_Ngrid200_new.pickle', 'rb') as data:
#	temp=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav100_ovr_egns_weight_dis1_d_rel_tsi_s12_s29_Ngrid200_horr_chk.pickle', 'rb') as data:
#	temp=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav100_ovr_egns_all_spin_echo_weight_dis1_d_rel_tsi_s12_s29_Ngrid200_horr_chk.pickle', 'rb') as data:
#	temp=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav100_all_spin_echo_weight_dis1_d_rel_tsi_s12_s29_Ngrid200_horr_chk.pickle', 'rb') as data:
#	temp=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/new/vipin/vipin_sample_N10_eav4_ovr_egns_spin_echo_weight_dis1_d_rel_tsi_s12_d6_Ngrid200_horr_chk.pickle', 'rb') as data:
	temp=pickle.load(data)


dt = temp[0]
D = temp[1]
s = temp[2]
#sample = 7
De = D.mean(axis=0)
sp = s.mean(axis=0)
#Deer = (De.mean(axis=0))

#er = np.std(De, axis=0)

#i = np.where(dt>=3.86*(10**13))[0]
i = range(200)
#print(Deer[i[0]])

#Deer = np.abs(Deer)
#I1 = simps(Deer[i], dt[i])

'''
I = np.zeros(9)
t_max = dt[i[-1]]-dt[i[0]]

I[0] = simps(Deer[0,0:200], dt[0:200])/(t_max)
I[1] = simps(Deer[1,0:200], dt[0:200])/(t_max)
I[2] = simps(Deer[2,0:200], dt[0:200])/(t_max)
I[3] = simps(Deer[3,0:200], dt[0:200])/(t_max)
I[4] = simps(Deer[4,0:200], dt[0:200])/(t_max)
'''

'''
I[0] = simps(Deer[sample,0], dt)/(t_max)
I[1] = simps(Deer[sample,1], dt)/(t_max)
I[2] = simps(Deer[sample,2], dt)/(t_max)
I[3] = simps(Deer[sample,3], dt)/(t_max)
I[4] = simps(Deer[sample,4], dt)/(t_max)
'''
#I[5] = simps(Deer[sample,5], dt)/(t_max)
#I[6] = simps(Deer[sample,6], dt)/(t_max)
#I[7] = simps(Deer[sample,7], dt)/(t_max)
#I[8] = simps(Deer[sample,8], dt)/(t_max)

#I = I1/(dt[i[-1]]-dt[i[0]])
#print(t_max)


'''
j = np.where(np.logical_and(dt>=4, dt<=10**2))[0]
popt_s, pcov_s = curve_fit(func, dt[j], Deer[4,j])
#plt.plot(dt, func(dt, *popt_s), '-', label=r"fit, r=8, onset = %f" %(popt_s[1]))
#plt.plot(dt,np.zeros(dt.size))
#plt.errorbar(dt, Deer,yerr=er, label=r"$r=8$")
'''






'''
J_typ = np.zeros(6)

j = np.where(np.logical_and(dt>=7821, dt<=131924))[0]
popt, pcov = curve_fit(func, dt[j], Deer[sample,3,j])
J_typ[0] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=3, onset = %f" %(popt[1]))

j = np.where(np.logical_and(dt>=8204, dt<=1.17726*10**6))[0]
popt, pcov = curve_fit(func, dt[j], Deer[sample,4,j])
J_typ[1] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=4, onset = %f" %(popt[1]))

j = np.where(np.logical_and(dt>=7880, dt<=9.33193*10**7))[0]
popt, pcov = curve_fit(func, dt[j], Deer[sample,5,j])
J_typ[2] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=5, onset = %f" %(popt[1]))

j = np.where(np.logical_and(dt>=54584, dt<=1.3044*10**11))[0]
popt, pcov = curve_fit(func, dt[j], Deer[sample,6,j])
J_typ[3] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=6, onset = %f" %(popt[1]))

j = np.where(np.logical_and(dt>=1.57897*10**7, dt<=6.4041*10**12))[0]
popt, pcov = curve_fit(func, dt[j], Deer[sample,7,j])
J_typ[4] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=7, onset = %f" %(popt[1]))

j = np.where(np.logical_and(dt>=1.75384*10**8, dt<=5.47956*10**13))[0]
popt, pcov = curve_fit(func, dt[j], Deer[sample,8,j])
J_typ[5] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=8, onset = %f" %(popt[1]))
s = np.array([4,5,6,7,8,9])
#plt.plot(s, 1/J_typ)
'''
#plt.plot(s, np.exp(-s/0.23))

#popt_j, pcov_j = curve_fit(fu, s, 1/J_typ)
#plt.plot(s, fu(s, *popt_j), '-', label=r"$e^{-r/\xi} fit, \xi = %f$" %(popt_j))
#print(popt_j)


'''
plt.plot(dt, Deer[0], label=r"$r=1$")
plt.plot(dt, Deer[1], label=r"$r=2$")
plt.plot(dt, Deer[2], label=r"$r=3$")
plt.plot(dt, Deer[3], label=r"$r=4$")
plt.plot(dt, Deer[4], label=r"$r=5$")
'''

'''
plt.plot(dt, Deer[sample,5], label=r"$r=6$")
plt.plot(dt, Deer[sample,6], label=r"$r=7$")
plt.plot(dt, Deer[sample,7], label=r"$r=8$")
plt.plot(dt, Deer[sample,8], label=r"$r=9$")
'''

#I = np.abs(I)


#plt.plot(dt, sp, label=r"$spin-echo$")
#plt.plot(dt, De[0], label=r"$r=0$")
#plt.plot(dt, De[1], label=r"$r=1$")
#plt.plot(dt, De[2], label=r"$r=2$")
#plt.plot(dt, De[3], label=r"$r=3$")
#plt.plot(dt, De[4], label=r"$r=4$")
#plt.plot(dt, De[5], label=r"$r=5$")
#plt.plot(dt, De[6], label=r"$r=6$")
#plt.plot(dt, np.zeros(dt.size))
#plt.plot(dt, D[1,1], label=r"$r=6$")
#plt.plot(dt, D[2], label=r"$r=4$")
#plt.plot(dt, D[3], label=r"$r=5$")


#j = np.where(np.logical_and(dt>=4, dt<=10**2))[0]
popt_s, pcov_s = curve_fit(func, dt, sp)
#plt.plot(dt, func(dt, *popt_s), '-', label=r"fit, spin-echo, onset = %f" %(popt_s[1]))

J_typ = np.zeros(4)

j = np.where(np.logical_and(dt>=1866, dt<=5855))[0]
popt, pcov = curve_fit(func, dt[j], De[3,j])
J_typ[0] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=3, onset = %f" %(popt[0]))

j = np.where(np.logical_and(dt>=1.43154*10**11, dt<=7.92388*10**11))[0]
popt, pcov = curve_fit(func, dt[j], De[4,j])
J_typ[1] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=4, onset = %f" %(popt[0]))

j = np.where(np.logical_and(dt>=7.44499*10**12, dt<=2.60841*10**13))[0]
popt, pcov = curve_fit(func, dt[j], De[5,j])
J_typ[2] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=5, onset = %f" %(popt[0]))

j = np.where(np.logical_and(dt>=3.92418*10**13, dt<=1.29987*10**14))[0]
popt, pcov = curve_fit(func, dt[j], De[6,j])
J_typ[3] = ((popt_s[1]-popt[1])/popt[0] + (-popt[1]/popt[0]))/2
#plt.plot(dt, func(dt, *popt), '-', label=r"fit, r=6, onset = %f" %(popt[0]))

s= np.array([3,4,5,6])
plt.plot(s, 1/J_typ)

popt_j, pcov_j = curve_fit(fu, s, 1/J_typ)
plt.plot(s, fu(s, *popt_j), '-', label=r"$e^{-r/\xi} fit, \xi = %f$" %(popt_j))

'''
plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$r=1$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$r=2$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$r=3$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$r=4$")
plt.errorbar(dt, Deer[4], yerr=er[4], label=r"$r=5$")
'''

'''
plt.plot(dt, np.full(dt.size, I[0]), label=r"$r=1, %f$" %(I[0]))
plt.plot(dt, np.full(dt.size, I[1]), label=r"$r=2, %f$" %(I[1]))
plt.plot(dt, np.full(dt.size, I[2]), label=r"$r=3, %f$" %(I[2]))
plt.plot(dt, np.full(dt.size, I[3]), label=r"$r=4, %f$" %(I[3]))
plt.plot(dt, np.full(dt.size, I[4]), label=r"$r=5, %f$" %(I[4]))
'''

'''
plt.plot(dt, np.full(dt.size, I[5]), label=r"$r=6, %f$" %(I[5]))
plt.plot(dt, np.full(dt.size, I[6]), label=r"$r=7, %f$" %(I[6]))
plt.plot(dt, np.full(dt.size, I[7]), label=r"$r=8, %f$" %(I[7]))
plt.plot(dt, np.full(dt.size, I[8]), label=r"$r=9, %f$" %(I[8]))
'''

#plt.plot(dt, np.zeros((dt.size)), label=r"$Zero$")
#plt.ion()
#plt.yticks([0,0.2,0.4,0.5])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.yscale('log')
#plt.xscale('log')
plt.legend()
plt.show()