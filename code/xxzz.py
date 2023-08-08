#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:55:23 2020

@author: abhishekraj
"""
import time
import numpy as np
import common as cm
import new_common as ncm
from numpy import linalg as la
from scipy import sparse as spr
import scipy.sparse.linalg as spla
from matplotlib import pyplot as plt

start_time = time.time()
#grid = 3
#Theta = np.linspace(0.0,np.pi/2,grid)
with open('/Users/abhishekraj/theta.txt', 'r') as f:
    data = f.read().splitlines() 
f.close()
Theta = np.array(list(map(float, data)))
grid = Theta.size
'''K = 1.0
delta = np.linspace(0.0,2.0,grid)'''
delta = np.sin(Theta)
K = np.cos(Theta)
nos = 1
gse = np.zeros((grid))
#diff = np.zeros((grid))
#gse = np.zeros((grid,nos))
fss = np.zeros(5)

def Eg_T(xlp,ylp):
    lx = xlp
    ly = ylp
    N = lx*ly
    print (lx,ly)
    
    for i in range(grid):
        print(i)
        #Ham = ncm.plising(delta[i],K[i],lx,ly) - 4*N*spr.identity(2**N)
        #Ham = cm.xx_t_zz(delta[i],K[i],lx,ly) - 4*N*spr.identity(2**N)
        #Ham = ncm.xx_t_zz(delta[i],K[i],lx,ly) - 4*N*spr.identity(2**N)
        Ham = ncm.xx_t_zz_obc(delta[i],K[i],lx,ly) - 4*N*spr.identity(2**N)
        e, v = spla.eigsh(Ham, k=nos)
        #e,v = la.eigh(Ham)
        e = e + 4.0*N
        '''k=0
        for j in range(nos):
            if np.allclose(e[j],e[0]):
                k=k+1
            gse[i,j] = (e[j])/(N)
        print(k)'''
        gse[i] = (e[0])/(N)
        '''if (i<=(grid+1)/2):
            gse[i] = (e[0])/(N-ly)
        else:
            gse[i] = (e[0])/(N-lx)'''
        '''if(i<=101):
            diff[i] = Emf_z[i]-gse[i]
        else:
            diff[i] = Emf_x[i]-gse[i]'''

    
    '''for i in range(nos):
        if(i==0):
            plt.plot(Theta,gse[:,i],'#008000')
        else:
            plt.plot(Theta,gse[:,i])'''
    
    if(lx==3):
        plt.plot(Theta,gse,'#008000',label=r"$n=$"+str(lx)+"x"+str(ly))
    else:
        plt.plot(Theta,gse,label=r"$n=$"+str(lx)+"x"+str(ly))
    #return gse


with open('/Users/abhishekraj/ene_mz.txt', 'r') as f:
    data = f.read().splitlines() 
f.close()
Emf_z = np.array(list(map(float, data)))

with open('/Users/abhishekraj/ene_mx.txt', 'r') as f:
    data = f.read().splitlines() 
f.close()
Emf_x = np.array(list(map(float, data)))
    
plt.plot(Theta,Emf_z,'--',label=r"$mf-z$")
plt.plot(Theta,Emf_x,'--',label=r"$mf-x$")


'''for i in range(3):
    fss[i] = Eg_T(i+3,3)[1]
    
fss[3] = Eg_T(4,4)[1]
fss[4] = Eg_T(5,4)[1]'''

Eg_T(3,3)
Eg_T(4,3)
Eg_T(5,3)
Eg_T(4,4)
Eg_T(5,4)


print("--- %s seconds ---" % (time.time() - start_time))
plt.xlabel(r"$\Theta$")
plt.ylabel(r"$E_g$")
#plt.ylabel(r"$E$")

#plt.hlines(Emf_z[101],9,20,'b','--',label=r"$mf$")

'''plt.scatter([9,12,15,16,20],Emf_z[101]-fss)
plt.xlabel(r"$N$")
#plt.ylabel(r"$E_g$")
plt.ylabel(r"$Emf-E_g$")'''

plt.legend()
#plt.savefig("Diff_theta.pdf")
plt.show()