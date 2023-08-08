import pickle
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
#from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x,mu,ss):
	return ((1./(np.sqrt(2.*np.pi)*ss))*np.exp(-np.log(np.abs(x-mu))**2/2.*ss**2))

#with open('/home/abhishek/Documents/ny project/data/hv5_N8_eav50_ovr_egns_dis100_t100000_d_rel_tsi_s14_Ngrid200_spa_corr.pickle', 'rb') as data:
#	temp=pickle.load(data)

with open('/home/abhishek/Documents/ny project/data/sig_plus_min_corr/hv10_N8_eav50_ovr_egns_dis100_t100000_d_rel_tsi_s14_Ngrid200_siz_plus_mins_corr.pickle', 'rb') as data:
	temp1=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/sig_plus_min_corr/hv10_N8_eav50_ovr_egns_dis1000_sig_plus_min_corr_s12_pal_huse.pickle', 'rb') as data:
#	temp1=pickle.load(data)
with open('/home/abhishek/Documents/ny project/data/sig_plus_min_corr/hv10_N8_eav50_dis100_t100000_d_rel_tsi_s14_rc_Ngrid200_siz_plus_mins_corr.pickle', 'rb') as data:
	temp=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/hv20_N8_eav50_ovr_egns_dis1000_spa_corr_pal_huse_all.pickle', 'rb') as data:
#	temp=pickle.load(data)

#with open('/home/abhishek/Documents/ny project/data/hv1_N8_eav50_ovr_egns_dis1000_spa_corr_s12_pal_huse.pickle', 'rb') as data:
#	temp1=pickle.load(data)


dt = temp[0]
D = np.abs(temp[1])
D_s = np.abs(temp1[1])
#s = np.arange(4)
#f = np.zeros((7))

'''
De = np.log(D.mean(axis=0))
Deer = De.mean(axis=0)
er = np.std(De, axis=0)

De1 = np.log(D1.mean(axis=0))
Deer1 = De1.mean(axis=0)
er1 = np.std(De1, axis=0)
'''


a=D.ravel()
i=np.where(a==0)[0]
a[i] = 10**(-30)
De=np.reshape(a,(100, 50, 4, 200))
#De=np.reshape(a,(1000, 50, 5))
Deer = np.log(De)

a_s=D_s.ravel()
i=np.where(a_s==0)[0]
a_s[i] = 10**(-30)
De_s=np.reshape(a_s,(100, 50, 4, 200))
#De_s=np.reshape(a_s,(1000, 50, 5))
Deer_s = np.log(De_s)
d2_s = Deer_s[:,:,2,:].ravel()
#d2_s = Deer_s[:,:,2].ravel()


d0 = Deer[:,:,0,10].ravel()
d1 = Deer[:,:,1,0:10].ravel()
d2 = Deer[:,:,2,0:10].ravel()
d3 = Deer[:,:,3,0:10].ravel()

d2_1 = Deer[:,:,2,10:50].ravel()
d2_2 = Deer[:,:,2,50:100].ravel()
d2_3 = Deer[:,:,2,100:150].ravel()

d0_n = Deer[:,:,0,:].ravel()
d1_n = Deer[:,:,1,:].ravel()
d2_n = Deer[:,:,2,150:200].ravel()
d3_n = Deer[:,:,3,:].ravel()


'''
d0 = Deer[:,:,0].ravel()
d1 = Deer[:,:,1].ravel()
d2 = Deer[:,:,2].ravel()
d3 = Deer[:,:,3].ravel()
d4 = Deer[:,:,4].ravel()
'''
'''
d0 = Deer[:,:,:,0].ravel()
d1 = Deer[:,:,0:5,1].ravel()
d2 = Deer[:,:,0:4,2].ravel()
d3 = Deer[:,:,0:3,3].ravel()
d4 = Deer[:,:,0:2,4].ravel()
d5 = Deer[:,:,0:1,5].ravel()
d6 = Deer[:,:,0,6].ravel()
'''

#De = D.mean(axis=0)
#Deer = De.mean(axis=0)

'''
d0 = (D[:,:,:,0].mean(axis=0)).mean(axis=0)
d1 = (D[:,:,0:5,1].mean(axis=0)).mean(axis=0)
d2 = (D[:,:,0:4,2].mean(axis=0)).mean(axis=0)
d3 = (D[:,:,0:3,3].mean(axis=0)).mean(axis=0)
d4 = (D[:,:,0:2,4].mean(axis=0)).mean(axis=0)
d5 = (D[:,:,0:1,5].mean(axis=0)).mean(axis=0)
d6 = (D[:,:,0,6].mean(axis=0)).mean(axis=0)


f[0] = d0.mean(axis=0)
f[1] = d1.mean(axis=0)
f[2] = d2.mean(axis=0)
f[3] = d3.mean(axis=0)
f[4] = d4.mean(axis=0)
f[5] = d5.mean(axis=0)
f[6] = d6

plt.plot(s,f)
'''

'''
#D1 = Deer[:,:,3]
#a = D1.ravel()
n1, bins1, patches1 = plt.hist(d1, bins='auto', normed='TRUE', histtype='step',label=r"$d=1, h_v=20$")
n2, bins2, patches2 = plt.hist(d2, bins='auto', normed='TRUE', histtype='step',label=r"$d=2, h_v=20$")
n3, bins3, patches3 = plt.hist(d3, bins='auto', normed='TRUE', histtype='step',label=r"$d=3, h_v=20$")
n4, bins4, patches4 = plt.hist(d4, bins='auto', normed='TRUE', histtype='step',label=r"$d=4, h_v=20$")
n5, bins5, patches5 = plt.hist(d5, bins='auto', normed='TRUE', histtype='step',label=r"$d=5, h_v=20$")
#n6, bins6, patches6 = plt.hist(d6, bins='auto', normed='TRUE', histtype='step',label=r"$d=6, h_v=15$")
'''
hist0, bin_edges0 = np.histogram(d0, bins='auto', density=True)
hist1, bin_edges1 = np.histogram(d1, bins='auto', density=True)
hist2, bin_edges2 = np.histogram(d2, bins='auto', density=True)
hist3, bin_edges3 = np.histogram(d3, bins='auto', density=True)

hist0_n, bin_edges0_n = np.histogram(d0_n, bins='auto', density=True)
hist1_n, bin_edges1_n = np.histogram(d1_n, bins='auto', density=True)
hist2_n, bin_edges2_n = np.histogram(d2_n, bins='auto', density=True)
hist3_n, bin_edges3_n = np.histogram(d3_n, bins='auto', density=True)



hist2_1, bin_edges2_1 = np.histogram(d2_1, bins='auto', density=True)
hist2_2, bin_edges2_2 = np.histogram(d2_2, bins='auto', density=True)
hist2_3, bin_edges2_3 = np.histogram(d2_3, bins='auto', density=True)

hist2_s, bin_edges2_s = np.histogram(d2_s, bins='auto', density=True)


#hist4, bin_edges4 = np.histogram(d4, bins='auto', density=True)
#hist5, bin_edges5 = np.histogram(d5, bins='auto', density=True)
#plt.bar(bin_edges[:-1], hist)

#plt.plot(bin_edges0[:-1], hist0,label=r"$d=0, h_v=10$")
#plt.plot(bin_edges1[:-1], hist1,label=r"$d=1, h_v=10$")
plt.plot(bin_edges2[:-1], hist2,label=r"$d=2, h_v=10,%.2f$" %(dt[10]))
#plt.plot(bin_edges3[:-1], hist3,label=r"$d=3, h_v=10$")

plt.plot(bin_edges2_1[:-1], hist2_1,label=r"$d=2, h_v=10,%.2f$" %(dt[50]))
plt.plot(bin_edges2_2[:-1], hist2_2,label=r"$d=2, h_v=10,%.2f$" %(dt[100]))
plt.plot(bin_edges2_3[:-1], hist2_3,label=r"$d=2, h_v=10,%.2f$" %(dt[150]))


#plt.plot(bin_edges0_n[:-1], hist0_n,label=r"$d=0, h_v=10, latter time$")
#plt.plot(bin_edges1_n[:-1], hist1_n,label=r"$d=1, h_v=10, latter time$")
plt.plot(bin_edges2_n[:-1], hist2_n,label=r"$d=2, h_v=10,%.2f$" %(dt[199]))
#plt.plot(bin_edges3_n[:-1], hist3_n,label=r"$d=3, h_v=10, latter time$")

plt.plot(bin_edges2_s[:-1], hist2_s,label=r"$d=2, static, h_v=10,%.2f$" %(dt[199]))

#plt.plot(bin_edges4[:-1], hist4,label=r"$d=4, h_v=10$")
#plt.plot(bin_edges5[:-1], hist5,label=r"$d=5, h_v=20$")
#plt.xlim(-40, max(bin_edges1))
#plt.show()


'''
i0 = np.where(hist0>=max(hist0)/2)[0]
i1 = np.where(hist1>=max(hist1)/2)[0]
i2 = np.where(hist2>=max(hist2)/2)[0]
i3 = np.where(hist3>=max(hist3)/2)[0]
i4 = np.where(hist4>=max(hist4)/2)[0]
i5 = np.where(hist5>=max(hist5)/2)[0]
w = np.zeros((6))
w[0] = bin_edges0[i0[-1]]-bin_edges0[i0[0]]
w[1] = bin_edges1[i1[-1]]-bin_edges1[i1[1]]
w[2] = bin_edges2[i2[-1]]-bin_edges2[i2[1]]
w[3] = bin_edges3[i3[-1]]-bin_edges3[i3[0]]
w[4] = bin_edges4[i4[-1]]-bin_edges4[i4[0]]
w[5] = bin_edges5[i5[-1]]-bin_edges5[i5[0]]

d=np.arange(6)
plt.plot( d, w, 'r--', label="h_v=20")
'''

'''
i2 = np.where(bin_edges2>-35)[0]
popt, pcov = curve_fit(func, bin_edges2[i2], hist2[i2-1])
print(popt)
plt.plot(bin_edges2[i2], func(bin_edges2[i2], *popt), 'o-', label='fit')
'''


'''
i1 = np.where(bin_edges1>-35)[0]
i2 = np.where(bin_edges2>-35)[0]
i3 = np.where(bin_edges3>-35)[0]
i4 = np.where(bin_edges4>-35)[0]
i5 = np.where(bin_edges5>-35)[0]
(mu1, sigma1) = norm.fit(hist1[i1-1])
(mu2, sigma2) = norm.fit(hist2[i2-1])
(mu3, sigma3) = norm.fit(hist3[i3-1])
print(mu3, mu2, mu1)
print(sigma3, sigma2, sigma1)


y = mlab.normpdf( bin_edges1[i1], mu1, sigma1)
l = plt.plot( bin_edges1[i1], y, 'r--', linewidth=2)
'''

'''
plt.errorbar(s, Deer1, yerr=er1, label=r"$h_v=1$")
plt.errorbar(s, Deer, yerr=er, label=r"$h_v=20$")
#plt.errorbar(dt, Deer, yerr=er, label=r"$DEER$")
'''

#plt.xticks([0,1,2,3])
#plt.yticks([0,0.2,0.4,0.5])
#plt.xlabel(r"$\langle\langle D(t) \rangle\rangle$", fontsize=18)
#plt.xlabel(r"$DEER-response$", fontsize=18)
#plt.xlabel(r"$\langle s_i^{z} s_j^{z} \rangle$", fontsize=18)

plt.xlabel(r"$ln|C_{n\alpha}^{+-}|$", fontsize=18)
plt.ylabel(r"$Frequency$", fontsize=18)
#plt.ylabel(r"$\sigma$", fontsize=18)
#plt.xlabel(r"$d$", fontsize=18)
#plt.ylabel(r"$P {[} \langle\langle D(t) \rangle\rangle ]$", fontsize=18)
#plt.xscale('log')
plt.legend()
plt.show()