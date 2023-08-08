import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('/home/abhishek/Documents/project/data/annni_hv1_N10_J0_NGrid_tau.pickle', 'rb') as data:
	temp=pickle.load(data)

with open('/home/abhishek/Documents/project/data/annni_hv01_N10_J01_NGrid_tau.pickle', 'rb') as data1:
	temp1=pickle.load(data1)

with open('/home/abhishek/Documents/project/data/annni_hv01_N10_J05_NGrid_tau.pickle', 'rb') as data2:
	temp2=pickle.load(data2)

with open('/home/abhishek/Documents/project/data/annni_hv01_N10_J1_NGrid_tau.pickle', 'rb') as data3:
	temp3=pickle.load(data3)

dt = temp[0]
dtau = temp[1]

mz = temp[2].mean(axis=1)
S_en = temp[3].mean(axis=1)
sc = temp[4].mean(axis=1)
l_fl = temp[5].mean(axis=1)
bi_fl = temp[6].mean(axis=1)

tau = 30

NGrid = temp[0].shape[0]
window = np.where(dtau>=tau)
c_tau = (window[0])[0]
print(dtau[c_tau])
c_dt = np.linspace(0,2.5*dtau[c_tau],NGrid)

avg_m_z = mz[c_tau,:]
S = S_en[c_tau,:]
avg_s_c = sc[c_tau,:]
l = l_fl[c_tau,:]
b_fl = bi_fl[c_tau,:]

S1 = temp1[3].mean(axis=1)[c_tau,:]
S2 = temp2[3].mean(axis=1)[c_tau,:]
S3 = temp3[3].mean(axis=1)[c_tau,:]


plt.plot(c_dt, S, label=r"$J'=0$")
plt.plot(c_dt, S1, label=r"$J'=0.1$")
plt.plot(c_dt, S2, label=r"$J'=0.5$")
plt.plot(c_dt, S3, label=r"$J'=1$")
#plt.xticks([50,100,150,200], " ")
#plt.yticks([0.6,0.4,0.2,0,-0.2])
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$S$", fontsize=18)
plt.legend()
plt.show()
'''
plt.subplot(5,1,1)
plt.plot(dt, avg_m_z)
plt.xticks([50,100,150,200], " ")
plt.yticks([0.6,0.4,0.2,0,-0.2])
plt.ylabel(r"$\langle m_z \rangle$", fontsize=18)
plt.subplot(5,1,2)
plt.plot(dt, S)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$S(t)$", fontsize=18)
plt.subplot(5,1,3)
plt.plot(dt, b_fl)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$\mathcal{F}$", fontsize=18)
plt.subplot(5,1,4)
plt.plot(dt, avg_s_c)
plt.xticks([50,100,150,200], " ")
plt.ylabel(r"$\langle s_c \rangle$", fontsize=18)
plt.subplot(5,1,5)
plt.plot(dt, l)
plt.xticks([0,50,100,150,200,250], [0,50,r"$\tau$",150,r"$2\tau$",250])
#plt.yticks([0,-0.2,-0.4,-0.6,-0.8,-1])
plt.ylabel(r"$l(t)$", fontsize=18)
plt.xlabel(r"$t$", fontsize=18)
plt.subplots_adjust(hspace=.4)
plt.show()'''