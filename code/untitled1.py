#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 01:44:03 2020

@author: abhishekraj
"""

import numpy as np
from matplotlib import pyplot as plt


with open('/Users/abhishekraj/theta.txt', 'r') as f:
    data = f.read().splitlines() 
f.close()
theta = np.array(list(map(float, data)))


with open('/Users/abhishekraj/psi_theta.txt', 'r') as f:
    data = f.read().splitlines() 
f.close()
psi = np.array(list(map(float, data)))

with open('/Users/abhishekraj/psi_th_x.txt', 'r') as f:
    data = f.read().splitlines() 
f.close()
psi1 = np.array(list(map(float, data)))
plt.plot(theta,psi,label=r"$mf-z$")
plt.plot(theta,psi1,label=r"$mf-x$")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\psi$")
plt.legend()
#plt.savefig("/Users/abhishekraj/Dropbox (Personal)/Plaquette/pl_ising_paper/plots/psi_delta.pdf")
plt.show()