import pickle
import numpy as np
from scipy import stats
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt

jp=[0,0.1,1]#np.linspace(0,1,40)
temp=[]
for j in jp:
	with open('/media/abhishek/705E9FF65E9FB378/Users/abhishekpc/Desktop/data/new/annni_hv01_N10_J'+str(j)+'_seed80_tau_energy_01.pickle', 'rb') as data:
		temp.append(pickle.load(data))

	#with open('/home/abhishek/Documents/project/data/new/N8/dh01/annni_hv01_N8_J'+str(j)+'_tau.pickle', 'rb') as data:
	#	temp.append(pickle.load(data))

	#with open('/home/abhishek/Documents/project/data/new/N6/dh01/new/annni_hv01_N6_J'+str(j)+'_tau.pickle', 'rb') as data:
	#	temp.append(pickle.load(data))

#rand_int_st/dh01/annni_dh01_ran_inst
#hv01/pbc/annni_hv01
#N6/dh01/annni_hv01_N6

#dt=temp[0][0]
jpgrid=3#40
taugrid=40
NGrid=300
dtau=temp[0][0]
dt = np.linspace(0,2.5*dtau[taugrid-1], NGrid)
window = (np.where(np.logical_and(dt>=1.75*dtau[39],dt<=2.25*dtau[39])))[0]

S = np.zeros((jpgrid,taugrid,window.size))
St = np.zeros((jpgrid,taugrid,window.size))
S_min = np.zeros((jpgrid,taugrid))
S_max = np.zeros((jpgrid,taugrid))
S_min_arg = np.zeros((jpgrid,taugrid), dtype='int')
std = np.zeros((jpgrid,taugrid))

for i in range(jpgrid):
	S[i,:,:] = temp[i][2].mean(axis=1)
	St[i,:,:] = np.std(temp[i][2], axis=1)
	S_max[i,:] = S[i,:,:].max(axis=-1)
	for j in range(taugrid):
		S_min[i,j] = S[i,j,window-210].min(axis=-1)
		S_min_arg[i,j] = np.argmin(S[i,j,window-210], axis=-1)
		std[i,j] = St[i,j,S_min_arg[i,j]]
		#std[i,j] = St[i,j,window[0]+S_min_arg[i,j]]

'''
z = 1 - (S_min/S_max)
plt.xlabel(r"$\tau$")
plt.ylabel(r"$1-\frac{S_{min}}{S_{max}}$")
plt.plot(dtau , z[10], label=r"$J'=%0.2f$" %jp[10])
plt.legend()
plt.show()
'''

'''
def tf_en(l):
	i = np.where(dt>=12)[0][0]
	j = np.where(dt>=dtau[l])[0][0]
	tem_en = np.zeros((40))
	for m in range(40):
		tem_en[m] = np.std(S[m,l,i:j])
	plt.xlabel(r"$J'$")
	plt.ylabel(r"$\sigma_t (\langle S \rangle)$")
	plt.plot(jp , tem_en, label=r"$\tau=%.2f$" %dtau[l])

def sd_en(e, k, l=-1):
	if l==-1:
		if e==0:
			tem_sig = S_min[k,:]
			lbl = r"$\langle S \rangle$"
		else:
			tem_sig = std[k,:]
			lbl = "std" 
		plt.plot(dtau , tem_sig, label=r"$J'=%.2f$" %jp[k])
		plt.xlabel(r"$\tau$")
		plt.ylabel(lbl)
		plt.xscale('log')
	else:
		if e==0:
			tem_sig = S[k,l,:]
			lbl = r"$\langle S \rangle$"
		else:
			tem_sig = St[k,l,:]
			lbl = "std" 		
		dt=np.linspace(0, 2.5*dtau[l], NGrid)
		plt.plot(dt , tem_sig, label=r"$J'=%.2f$" %jp[k])
		plt.xticks([0,dtau[l],1.75*dtau[l],2*dtau[l],2.25*dtau[l]], [0,r"$\tau$","{0:.2f}".format(1.75*dtau[l]),r"$2\tau$","{0:.2f}".format(2.25*dtau[l])])
		plt.xlabel(r"$t$")
		plt.ylabel(lbl)


#spec = int(input("enter 0 for mean of entropy and 1 for sd of entropy: "))
#jp_val = int(input("enter the index for J': "))
tau_val = int(input("enter the index for tau: "))
tf_en(tau_val)
#sd_en(spec, jp_val, tau_val)
#sd_en(1, jp_val, tau_val)
plt.legend()
plt.show()
'''


'''
z = 1 - (S_min/S_max)
sol = np.zeros((jpgrid))
for i in range(jpgrid):
	f = interpolate.interp1d(dtau, z[i,:])
	g = lambda x: f(x)-0.5
	sol[i] = optimize.brentq(g , dtau.min(), dtau.max())



with open('/home/abhishek/Documents/project/data/new/tau_at_0.5_Jp_dh01_hv01_N6', 'wb') as data:
	pickle.dump(sol, data)

plt.plot(jp, sol)
plt.xlabel(r"$J'$", fontsize=20)
plt.ylabel(r"$\tau$", fontsize=20)
#plt.legend()
plt.show()
'''


'''
e = stats.sem(temp[3], axis=1)
e1 = stats.sem(temp1[3], axis=1)
e2 = stats.sem(temp2[3], axis=1)
e3 = stats.sem(temp3[3], axis=1)

ye_min = np.zeros((dtau.shape[0]))
ye_min1 = np.zeros(dtau.shape[0])
ye_min2 = np.zeros(dtau.shape[0])
ye_min3 = np.zeros(dtau.shape[0])
ye_max = np.zeros((dtau.shape[0]))
ye_max1 = np.zeros(dtau.shape[0])
ye_max2 = np.zeros(dtau.shape[0])
ye_max3 = np.zeros(dtau.shape[0])

amin = S[:,window].argmin(axis=1)
amin1 = S1[:,window].argmin(axis=1)
amin2 = S2[:,window].argmin(axis=1)
amin3 = S3[:,window].argmin(axis=1)
amax = S.argmax(axis=1)
amax1 = S1.argmax(axis=1)
amax2 = S2.argmax(axis=1)
amax3 = S3.argmax(axis=1)
for i in range(dtau.shape[0]):
	ye_min[i]=e[i,window[0]+amin[i]]
	ye_min1[i]=e1[i,window[0]+amin1[i]]
	ye_min2[i]=e2[i,window[0]+amin2[i]]
	ye_min3[i]=e3[i,window[0]+amin3[i]]
	ye_max[i]=e[i,amax[i]]
	ye_max1[i]=e1[i,amax1[i]]
	ye_max2[i]=e2[i,amax2[i]]
	ye_max3[i]=e3[i,amax3[i]]



#ne=np.sqrt((ye_min/S_min)**2+(ye_max/S_max)**2)*(S_min/S_max)
#ne1=np.sqrt((ye_min1/S_min1)**2+(ye_max1/S_max1)**2)*(S_min1/S_max1)
#ne2=np.sqrt((ye_min2/S_min2)**2+(ye_max2/S_max2)**2)*(S_min2/S_max2)
#ne3=np.sqrt((ye_min3/S_min3)**2+(ye_max3/S_max3)**2)*(S_min3/S_max3)


ne=((ye_min/S_min)+(ye_max/S_max))*(S_min/S_max)
ne1=((ye_min1/S_min1)+(ye_max1/S_max1))*(S_min1/S_max1)
ne2=((ye_min2/S_min2)+(ye_max2/S_max2))*(S_min2/S_max2)
ne3=((ye_min3/S_min3)+(ye_max3/S_max3))*(S_min3/S_max3)
'''

'''
X,Y = np.meshgrid(dtau, jp)
fig, ax = plt.subplots()
ax.set_rasterized(True)
p = ax.pcolormesh(X, Y, ( 1 - (S_min/S_max)), cmap='jet')
cb = fig.colorbar(p)
cb.set_label(label=r"$1-\frac{S_min}{S_max}$",fontsize=18)
#cb.set_label(label=r"$\frac{S_{min}}{S_{max}}$",fontsize=20)
#plt.plot(h,jp)
'''
plt.plot(dtau,  1-(S_min[0]/S_max[0]), label=r"$J'=0$")
plt.plot(dtau,  1-(S_min[1]/S_max[1]), label=r"$J'=0.1$")
plt.plot(dtau,  1-(S_min[2]/S_max[2]), label=r"$J'=1$")
#plt.errorbar(dtau, S_min, yerr=ye_min, label=r"$J'=0$")
#plt.errorbar(dtau, S_min1, yerr=ye_min1, label=r"$J'=0.1$")
#plt.errorbar(dtau, S_min2, yerr=ye_min2, label=r"$J'=0.5$")
#plt.errorbar(dtau, S_min3, yerr=ye_min3, label=r"$J'=1$")
#plt.xticks([50,100,150,200], " ")
#plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel(r"$\tau$", fontsize=20)
#plt.ylabel(r"$J'$", fontsize=20)
#plt.ylabel(r"$S_{min}$", fontsize=18)
plt.ylabel(r"$(S_{max}-S_{min})/S_{max}$", fontsize=18)
#plt.ylabel(r"$(\mathcal{F}_{max}-\mathcal{F}_{min})/\mathcal{F}_{max}$", fontsize=18)
plt.xscale('log')
plt.legend()
plt.show()
