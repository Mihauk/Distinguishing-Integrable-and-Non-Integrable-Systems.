import pickle
import numpy as np
from scipy import stats
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

with open('/home/abhishek/Documents/ny project/data/hv10_N8_eav50_dis100_t10000_d_rel_tsi_s14_rc_Ngrid200.pickle', 'rb') as data:
	temp=pickle.load(data)

dt = temp[0]
D = temp[1]

De = D.mean(axis=0)
Deer = De.mean(axis=0)
N = Deer[0].size

k = np.logspace(0, ((2*np.pi)/(dt[50]-dt[49]))**0.1, (dt[150]-dt[149])/(dt[50]-dt[49]))
Deer_fft1 = np.zeros((k.size),dtype="complex")
Deer_fft2 = np.zeros((k.size),dtype="complex")
Deer_fft3 = np.zeros((k.size),dtype="complex")
for i in range(k.size):
	print(i)
	for j in range(N):
		Deer_fft1[i] += np.exp(-1j*k[i]*dt[j])*Deer[1,j]
		Deer_fft2[i] += np.exp(-1j*k[i]*dt[j])*Deer[2,j]
		Deer_fft3[i] += np.exp(-1j*k[i]*dt[j])*Deer[3,j]


'''
y = fft(Deer[0])
f = fftfreq(Deer[0].size, dt[1]-dt[0])
'''

#plt.plot(f[0:N//2], y[0:N//2].real)
plt.plot(k, np.abs(Deer_fft1), label="d=1")
plt.plot(k, np.abs(Deer_fft2), label="d=2")
plt.plot(k, np.abs(Deer_fft3), label="d=3")
plt.xlabel(r"$J$", fontsize=18)
plt.ylabel(r"$ P(J) $", fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()