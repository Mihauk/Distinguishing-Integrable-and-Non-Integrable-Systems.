import pickle
import numpy as np
from scipy import stats
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt

jp=np.linspace(0,1,40)
#temp=[]

with open('/media/abhishek/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/N6/sample103/annni_dh01_103sample_ran_inst_N6_J0.0_tau.pickle', 'rb') as data:
	temp = (pickle.load(data))

dt = temp[0]
jpgrid = 40
taugrid = 40
NGrid = 300
dtau = temp[1]
ene = temp[7]
s = temp[3]
tau = 39
sample = s.shape[1]

window = (np.where(np.logical_and(dt>=1.75*dtau[39],dt<=2.25*dtau[39])))[0]
del_e = np.zeros(sample)
s_min = np.zeros(sample)

for i in range(sample):
	del_e[i] = ene[tau, i, 299] + ene[tau, i, 0]
	s_min[i] = s[tau, i, window].min()


#plt.clf()
#plt.ion()
plt.plot(del_e+np.random.randn(1000)*.001, s_min, '.')
plt.show()