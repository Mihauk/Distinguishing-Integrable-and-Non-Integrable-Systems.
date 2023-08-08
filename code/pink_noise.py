import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def pink_dis(c,n):
	phi = 2*np.pi*np.random.rand(n)
	phi_t = np.exp(1j*phi)
	#phi_t = np.cos(phi) + 1j*np.sin(phi)
	c[0:n] *= phi_t
	c[-1:-n-1:-1] = np.conj(c[0:n])

	y = np.fft.ifft(c).real
	return y


N = 200
l = 50 #max_length
samples = 1000

k = np.linspace(-l,l,N)
f = (k*l)/(2*np.pi)

#assert N%2!=0
#c = np.zeros(N,dtype='complex')
c = 1/np.abs(f)
c = c + 0j
n = int((N-1)/2)
c[n] = 0j

assert N%2==0
o_f = np.zeros((samples, int(N/2)+1))

for i in range(samples):

	t = pink_dis(c,n)

	p_f = f[n:]
	p_c = c[n:]
	p_t = t[n:]

	'''
	plt.plot(p_f, p_c.real)
	#plt.plot(p_f, p_t)
	#plt.xticks([50,100,150,200], " ")
	#plt.yticks([0.6,0.4,0.2,0,-0.2])
	plt.xlabel(r"$f$", fontsize=18)
	#plt.xlabel(r"$t$", fontsize=18)
	#plt.ylabel(r"$$", fontsize=18)
	#plt.legend()
	plt.show()'''

	m1 = st.moment(t,1)
	m2 = st.moment(t,2)
	m3 = st.moment(t,3)
	m4 = st.moment(t,4)

	'''
	print(m1)
	print(m2)
	print(m3)
	print(m4)
	'''

	k1 = m1
	k2 = m2 - (m1**2)
	k3 = m3 - 3*m2*m1 + 2*(m1**3)
	k4 = m4 - 4*m3*m1 - 3*(m2**2) + 12*m2*(m1**2) -6*(m1**4)


	o = np.exp(1j*k1*p_f - .5*k2*(p_f**2) - (4j/3)*k3*(p_f**3) + (2/3)*k4*(p_f**4))
	o_f[i,:] = .5*(1+o.real)


o_f_av = o_f.mean(axis=0)

plt.plot(p_f, o_f_av)
#plt.plot(p_f, p_t)
#plt.xticks([50,100,150,200], " ")
#plt.yticks([0.6,0.4,0.2,0,-0.2])
#plt.xlabel(r"$f$", fontsize=18)
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"$\langle D(t) \rangle$", fontsize=18)
#plt.legend()
plt.show()
