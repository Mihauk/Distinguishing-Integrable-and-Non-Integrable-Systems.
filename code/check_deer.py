import sys
import pickle
import numpy as np
import scipy as sc
import common as cm
from scipy import integrate
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq

with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_dis100_t100000_d_rel_tsi_s14_rc_Ngrid200.pickle', 'rb') as data:
	temp=pickle.load(data)

dt = temp[0]
D = temp[1]

De = D.mean(axis=0)
Deer = De.mean(axis=0)

er = np.std(De, axis=0)

def PJ(J,J_ty,c,r):
	return (np.exp(-(np.log(np.abs(J/J_ty)))**2/(c*r))/(np.abs(J)*np.sqrt(c*r*np.pi)))

'''
P_J = lambda x,J_ty,c,r : (np.exp(-(np.log(np.abs(x/J_ty)))**2/(c*r))/(np.abs(x)*np.sqrt(c*r*np.pi)))
r = 5
z = 1
c = 5
J_ty = np.exp(-r/z)

y, err = integrate.quad(P_J, 0, 3*J_ty, args=(J_ty,c,r,))
print(y)
'''


x = 1
a = 1
c = 10
r = np.linspace(1,4,4)
J_ty = a*np.exp(-r/x)
J = np.linspace(0.001,0.77,100000)
P_J = np.zeros((r.size, J.size))
y = np.zeros((r.size, J.size))
N = J.size

for j in range(r.size):
	for i in range(J.size):
		P_J[j, i] = PJ(J[i],J_ty[j],c,r[j])
	y[j,:] = ifft(P_J[j,:])

f = fftfreq(J.size, J[1]-J[0])

#plt.plot(J, P_J[0])
plt.errorbar(dt, Deer[0], yerr=er[0], label=r"$r=1$")
plt.errorbar(dt, Deer[1], yerr=er[1], label=r"$r=2$")
plt.errorbar(dt, Deer[2], yerr=er[2], label=r"$r=3$")
plt.errorbar(dt, Deer[3], yerr=er[3], label=r"$r=4$")
plt.plot(f[0:N//2], y[0,0:N//2], label=r"$r=1, fft$")
plt.plot(f[0:N//2], y[1,0:N//2], label=r"$r=2, fft$")
plt.plot(f[0:N//2], y[2,0:N//2], label=r"$r=3, fft$")
plt.plot(f[0:N//2], y[3,0:N//2], label=r"$r=4, fft$")
plt.xscale('log')
plt.legend()
plt.show()