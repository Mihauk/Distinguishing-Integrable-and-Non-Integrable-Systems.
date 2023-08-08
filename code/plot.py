import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

with open('/home/abhishek/Documents/project/data/annni_dh001_N10_J01_eff.pickle', 'rb') as data:
	temp=pickle.load(data)



with open('/home/abhishek/Documents/project/data/annni_dh001_N10_J05_eff.pickle', 'rb') as data1:
	temp1=pickle.load(data1)

with open('/home/abhishek/Documents/project/data/annni_dh001_N10_J1_eff.pickle', 'rb') as data2:
	temp2=pickle.load(data2)



with open('/home/abhishek/Documents/project/data/annni_dh005_N10_J01_eff.pickle', 'rb') as data3:
	temp3=pickle.load(data3)

with open('/home/abhishek/Documents/project/data/annni_dh005_N10_J05_eff.pickle', 'rb') as data4:
	temp4=pickle.load(data4)

with open('/home/abhishek/Documents/project/data/annni_dh005_N10_J1_eff.pickle', 'rb') as data5:
	temp5=pickle.load(data5)



with open('/home/abhishek/Documents/project/data/annni_hv01_N10_J01_eff.pickle', 'rb') as data6:
	temp6=pickle.load(data6)

with open('/home/abhishek/Documents/project/data/annni_hv01_N10_J05_eff.pickle', 'rb') as data7:
	temp7=pickle.load(data7)

with open('/home/abhishek/Documents/project/data/annni_hv01_N10_J1_eff.pickle', 'rb') as data8:
	temp8=pickle.load(data8)


S1=temp1[2].mean(axis=0)

S2=temp2[2].mean(axis=0)

S3=temp3[2].mean(axis=0)
S4=temp4[2].mean(axis=0)
S5=temp5[2].mean(axis=0)

S6=temp6[2].mean(axis=0)
S7=temp7[2].mean(axis=0)
S8=temp8[2].mean(axis=0)


Jp = np.array([0.1,0.5,1])
S_min = np.zeros(3)
S_min1 = np.zeros(3)
S_min2 = np.zeros(3)

dt=temp[0]
mz=temp[1]
S_en=temp[2]
sc=temp[3]
l_fl=temp[4]
bi_fl=temp[5]

avg_m_z = mz.mean(axis=0)
S = S_en.mean(axis=0)
avg_s_c = sc.mean(axis=0)
l = l_fl.mean(axis=0)
b_fl = bi_fl.mean(axis=0)


tau=100
window = (np.where(np.logical_and(dt>=1.75*tau,dt<=2.25*tau)))[0]

S_min[0] = S[window].min()
S_min[1] = S1[window].min()
S_min[2] = S2[window].min()
S_min1[0] = S3[window].min()
S_min1[1] = S4[window].min()
S_min1[2] = S5[window].min()
S_min2[0] = S6[window].min()
S_min2[1] = S7[window].min()
S_min2[2] = S8[window].min()


e = stats.sem(temp[2], axis=0)
e1 = stats.sem(temp1[2], axis=0)
e2 = stats.sem(temp2[2], axis=0)

e3 = stats.sem(temp3[2], axis=0)
e4 = stats.sem(temp4[2], axis=0)
e5 = stats.sem(temp5[2], axis=0)

e6 = stats.sem(temp6[2], axis=0)
e7 = stats.sem(temp7[2], axis=0)
e8 = stats.sem(temp8[2], axis=0)

yer = np.zeros(3)
yer[0] = e[window[0]+S[window].argmin()]
yer[1] = e1[window[0]+S1[window].argmin()]
yer[2] = e2[window[0]+S2[window].argmin()]

yer1 = np.zeros(3)
yer1[0] = e3[window[0]+S3[window].argmin()]
yer1[1] = e4[window[0]+S4[window].argmin()]
yer1[2] = e5[window[0]+S5[window].argmin()]

yer2 = np.zeros(3)
yer2[0] = e6[window[0]+S6[window].argmin()]
yer2[1] = e7[window[0]+S7[window].argmin()]
yer2[2] = e8[window[0]+S8[window].argmin()]

plt.scatter(Jp, S_min2, label=r"$\delta h=0.1$")
plt.scatter(Jp, S_min1, label=r"$\delta h=0.05$")
plt.scatter(Jp, S_min, label=r"$\delta h=0.01$")
#plt.errorbar(Jp, S_min2, yerr=yer2, label=r"$\delta h=0.1$")
#plt.errorbar(Jp, S_min1, yerr=yer1, label=r"$\delta h=0.05$")
#plt.errorbar(Jp, S_min, yerr=yer, label=r"$\delta h=0.01$")
plt.legend()
plt.xlabel(r"$J'$", fontsize=18)
plt.ylabel(r"$S_{min}$", fontsize=18)
plt.show()
