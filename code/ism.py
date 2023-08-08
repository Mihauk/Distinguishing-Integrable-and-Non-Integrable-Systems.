import numpy as np
import spin as s

def hamiltonian(h, J, N):
	H=0
	for i in range(N):
		H = H - J*np.dot(s.spin("Z",i,N),s.spin("Z",(i+1)%N,N)) + h*s.spin("X",i,N) - 1e-3*s.spin("Z",i,N)
	return(H)

def magnetization(N):
	mz=0
	for i in range(N):
		mz = mz + (1./N)*s.spin("Z",i,N)
	return(mz)