#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:18:52 2020

@author: abhishekraj
"""

import numpy as np
from scipy import sparse as spr

up = spr.csr_matrix(np.array([1, 0]))
down = spr.csr_matrix(np.array([0, 1]))
s_0 = spr.csr_matrix(np.array([[1, 0],[0, 1]]))
s_x = spr.csr_matrix(np.array([[0, 1],[1, 0]]))
s_y = spr.csr_matrix(np.array([[0, -1j],[1j, 0]]))
s_z = spr.csr_matrix(np.array([[1, 0],[0, -1]]))

s_plus = s_x + 1j*s_y
s_minus = s_x - 1j*s_y

s1_x= spr.csr_matrix(1/np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
s1_y = spr.csr_matrix(1/(1j*np.sqrt(2))*np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]))
s1_z = spr.csr_matrix(np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]))

s1_plus = s1_x + 1j*s1_y
s1_minus = s1_x - 1j*s1_y

def spin(A, i, N):
	assert (i<N), "specify the boundary condition"
	assert (A=="X" or A=="Y" or A=="Z" or A=="I" or A=="+" or A=="-"), "Specify the correct pauli matrices"
	if A=='X':
		sp=s_x
	elif A=='Y':
		sp=s_y
	elif A=='Z':
		sp=s_z
	elif A=='+':
		sp=s_plus
	elif A=='-':
		sp=s_minus
	s=spr.kron(spr.identity(2**i),spr.kron(sp,spr.identity(2**(N-i-1))))
	return s

def plising(h,k,lx,ly):
    assert (lx>=ly), "keep the X-direction larger"
    N = lx*ly
    w = np.arange(N)
    x = w%lx
    y = w//lx
    tx = (x+1)%lx + lx*y
    ty = x + lx*((y+1)%ly)
    ty1 = np.concatenate((tx[lx:(N)],tx[0:lx]),axis=None)
    H = 0
    for i in range(N):
        H = H - k*(spin("Z",i,N) @ spin("Z",tx[i],N) @ spin("Z",ty[i],N) @ spin("Z",ty1[i],N)) - h*spin("X",i,N)
    return (H)

def xx_t_zz(h,k,lx,ly):
    assert (lx>=ly), "keep the X-direction larger"
    N = lx*ly
    w = np.arange(N)
    x = w%lx
    y = w//lx
    tx = (x+1)%lx + lx*y
    ty = x + lx*((y+1)%ly)
    H = 0
    for i in range(N):
         H = H - k*(spin("Z",i,N) @ spin("Z",tx[i],N)) - h*(spin("X",i,N) @ spin("X",ty[i],N))
    return (H)

def xx_t_zz_obc(h,k,lx,ly):
    assert (lx>=ly), "keep the X-direction larger"
    N = lx*ly
    w = np.arange(N)
    x = w%lx
    y = w//lx
    tx = (x+1)%lx + lx*y
    ty = x + lx*((y+1)%ly)
    H = 0
    for i in range(N-1):
        if (N-i)<=lx:
            H = H - k*(spin("Z",i,N) @ spin("Z",tx[i],N))
        elif (i+1)%lx==0: 
            H = H - h*(spin("X",i,N) @ spin("X",ty[i],N))
        else:
            H = H - k*(spin("Z",i,N) @ spin("Z",tx[i],N)) - h*(spin("X",i,N) @ spin("X",ty[i],N))
    return (H)
