import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

with open('/home/abhishek/Documents/project/data/tfim_hv1_N10_J11.pickle', 'rb') as data:
	temp=pickle.load(data)

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


plt.subplot(5,1,1)
plt.plot(dt, avg_m_z)
plt.xticks([50,100,150,200], " ")
#plt.yticks([0.6,0.4,0.2,0,-0.2])
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
plt.show()