import numpy as np
from numpy import linalg as la

up = np.array([1, 0])
down = np.array([0, 1])
s_0 = np.array([[1, 0],[0, 1]])
s_x = .5*np.array([[0, 1],[1, 0]])
s_y = .5*np.array([[0, -1j],[1j, 0]])
s_z = .5*np.array([[1, 0],[0, -1]])
s_plus = s_x + 1j*s_y
s_minus = s_x - 1j*s_y

def spin(A, i, N):
	assert (i<N), "specify the boundary condition"
	assert (A=="X" or A=="Y" or A=="Z" or A=="I" or A=="+" or A=="-"), "Specify the corect pauli spin matrices from X, Y and Z"
	s=1
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
	elif A=='I':
		sp=s_0
	for ii in range(N):
		if i==ii:
			s=np.kron(s,sp)
		else :
			s=np.kron(s,s_0)
	return s

def h_tfim(h, J, N, h_v=None):
	if h_v is None:
		h_v = np.zeros((N))
	H = 0
	for i in range(N):
		H = H - J*np.dot(spin("Z",i,N),spin("Z",(i+1)%N,N)) + h*spin("X",i,N) + h_v[i]*spin("Z",i,N) #- 1e-3*spin("Z",i,N)
	return(H)

def h_xxz(h, J, N, d=0, h_v=None):
	if h_v is None:
		h_v = np.zeros((N))
	H = 0
	for i in range(N):
		H = H - J*(np.dot(spin("X",i,N),spin("X",(i+1)%N,N)) + np.dot(spin("Y",i,N),spin("Y",(i+1)%N,N)) + d*np.dot(spin("Z",i,N),spin("Z",(i+1)%N,N))) + h*spin("Z",i,N) + h_v[i]*spin("Z",i,N) #- 1e-3*spin("Z",i,N)
	return(H)

def h_xxz_obc(h, J, N, d=0, h_v=None):
	if h_v is None:
		h_v = np.zeros((N))
	H = 0
	for i in range(N-1):
		H = H - J*(np.dot(spin("+",i,N),spin("-",(i+1),N)) + np.dot(spin("-",i,N),spin("+",(i+1),N)) + d*np.dot(spin("Z",i,N),spin("Z",(i+1),N))) + h*spin("Z",i,N) + h_v[i]*spin("Z",i,N) #- 1e-3*spin("Z",i,N)
	return(H)

def h_annni(h, J, J_prime, N, h_v=None):
	if h_v is None:
		h_v = np.zeros((N))
	H = 0
	for i in range(N):
		H = H - J*np.dot(spin("Z",i,N),spin("Z",(i+1)%N,N)) + h*spin("X",i,N) + h_v[i]*spin("Z",i,N) + J_prime*np.dot(spin("Z",i,N),spin("Z",(i+2)%N,N)) #- 1e-3*spin("Z",i,N)
	return(H)

def h_annni_obc(h, J, J_prime, N, h_v=None):
	if h_v is None:
		h_v = np.zeros((N))
	H = 0
	for i in range(N-1):
		H = H - J*np.dot(spin("Z",i,N),spin("Z",(i+1)%N,N)) + h*spin("X",i,N) + h_v[i]*spin("Z",i,N) + J_prime*np.dot(spin("Z",i,N),spin("Z",(i+2)%N,N)) #- 1e-3*spin("Z",i,N)
	H = H + h_v[N-1]*spin("Z",N-1,N)
	return(H)

def h_rfh( J, N, h_v):
	H=0
	for i in range(N):
		H = H + J*(np.dot(spin("X",i,N),spin("X",(i+1)%N,N)) + np.dot(spin("Y",i,N),spin("Y",(i+1)%N,N)) + np.dot(spin("Z",i,N),spin("Z",(i+1)%N,N))) + h_v[i]*spin("Z",i,N)
	return (H)

def h_rfh_obc( J, N, h_v=None):
	if h_v is None:
		h_v = np.zeros((N))
	H=0
	for i in range(N-1):
		H = H + J*(np.dot(spin("X",i,N),spin("X",(i+1),N)) + np.dot(spin("Y",i,N),spin("Y",(i+1),N)) + np.dot(spin("Z",i,N),spin("Z",(i+1),N))) + h_v[i]*spin("Z",i,N)
	H = H + h_v[N-1]*spin("Z",N-1,N)
	return (H)

def h_rfxxz( J, delta, N, h_v):
	H=0
	for i in range(N):
		H = H + J*(np.dot(spin("X",i,N),spin("X",(i+1)%N,N)) + np.dot(spin("Y",i,N),spin("Y",(i+1)%N,N))) + delta*(np.dot(spin("Z",i,N),spin("Z",(i+1)%N,N))) + h_v[i]*spin("Z",i,N)
	return (H)

def h_rfxxz_obc( J, delta, N, h_v):
	H=0
	for i in range(N-1):
		H = H + J*(np.dot(spin("X",i,N),spin("X",(i+1),N)) + np.dot(spin("Y",i,N),spin("Y",(i+1),N))) + delta*(np.dot(spin("Z",i,N),spin("Z",(i+1),N))) + h_v[i]*spin("Z",i,N)
	H = H + h_v[N-1]*spin("Z",N-1,N)
	return (H)

def sl2(N):
	s_l2 = 0
	assert (N%2 == 0)
	for i in range(int(N/2)):
		s_l2 = s_l2 + spin("Z",i,N)
	return(s_l2)

def magnetization(N):
	mz = 0
	for i in range(N):
		mz = mz + (1./N)*spin("Z",i,N)
	return(mz)

def entropy(psi_t,r):
	S = 0
	m = np.reshape(psi_t, (r,r))
	u,s,v = la.svd(m)
	s1=s**2
	S = -np.dot(s1,np.log(s1))	
	x = check_real(S)
	return(x)

def bipartite_fl(s_l2, psi_t, psi_t_dagg):
	psi_p=np.dot(s_l2,psi_t)
	psi_p_dagg=np.conjugate(np.transpose(psi_p))
	f = np.dot(psi_p_dagg,psi_p) - (np.dot(psi_t_dagg,psi_p))**2
	x = check_real(f)
	return(x)

def spin_corr(N):
	sc=0
	for i in range(N):
		sc = sc + np.dot(spin("Z",i,N),spin("Z",(i+2)%N,N))
	return(sc)

def fidelity(psi_0,psi_t,N):
	l = np.log(np.abs(np.dot(psi_0,psi_t))**2)/N
	x = check_real(l)
	return(x)

def hermitian(x):
	y=x
	y=np.conjugate(np.transpose(y))
	if np.allclose(x,y):
		print("Hermitian")
	else:
		print("Not Hermitian")

def exp_m(e,v,v_dagg,t):
	exp = np.dot(v, np.dot(np.diag(np.exp(-1j*e*t)), v_dagg))
	return(exp)

def psi_at_t(e,v,v_dagg,t,psi_0):
	psi_t = np.dot(v,np.dot(np.diag(np.exp(-1j*e*t)),np.dot(v_dagg,psi_0)))
	return(psi_t)

def psi_0_xxz(N):
	s = 1
	for ii in range(N):
		if ii%2==0:
			s = np.kron(s,up)
		else:
			s = np.kron(s,down)
	return (s)

def rnd_egnst(sample, N):
	st = np.zeros((sample, 2**N))
	for jj in range(sample):
		s = 1
		r = np.random.randint(2, size=N)
		for ii in range(N):
			if r[ii] == 1:
				s = np.kron(s,up)
			else:
				s = np.kron(s,down)
		st[jj] = s
	return (st)

def check_real(x):
	temp = x
	assert (np.allclose(temp.imag, 0))
	return (temp.real)