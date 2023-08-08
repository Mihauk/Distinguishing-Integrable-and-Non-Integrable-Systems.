import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('/home/abhishek/Documents/project/data/new/tau_at_0.5_Jp_dh01_hv01_N10', 'rb') as data:
	temp=pickle.load(data)

with open('/home/abhishek/Documents/project/data/new/tau_at_0.5_Jp_dh01_hv01_N8', 'rb') as data:
	temp1=pickle.load(data)

with open('/home/abhishek/Documents/project/data/new/tau_at_0.5_Jp_dh01_hv01_N6', 'rb') as data:
	temp2=pickle.load(data)

jp=np.linspace(0,1,40)

plt.plot(jp, temp, label=r"$N=10$")
plt.plot(jp, temp1, label=r"$N=8$")
plt.plot(jp, temp2, label=r"$N=6$")

plt.ylabel(r"$\tau$", fontsize=18)
plt.xlabel(r"$J'$", fontsize=18)
plt.legend()
plt.show()