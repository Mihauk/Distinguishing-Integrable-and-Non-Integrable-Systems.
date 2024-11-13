import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla

# Define the spin operators as sparse matrices
up = np.array([1, 0], dtype=complex)
down = np.array([0, 1], dtype=complex)

s_0 = sp.csr_matrix([[1, 0], [0, 1]], dtype=complex)
s_x = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
s_y = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
s_z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)

s_plus = s_x + 1j * s_y
s_minus = s_x - 1j * s_y

s1_x = (1 / np.sqrt(2)) * sp.csr_matrix(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
s1_y = (1 / (1j * np.sqrt(2))) * sp.csr_matrix(
    [[0, 1, 0], [-1, 0, 1], [0, -1, 0]], dtype=complex)
s1_z = sp.csr_matrix([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)

s1_plus = s1_x + 1j * s1_y
s1_minus = s1_x - 1j * s1_y


def spin(A, i, N):
    assert i < N, "Specify the boundary condition"
    assert A in ["X", "Y", "Z", "I", "+", "-"], "Invalid Pauli matrix"

    if A == 'X':
        spm = s_x
    elif A == 'Y':
        spm = s_y
    elif A == 'Z':
        spm = s_z
    elif A == '+':
        spm = s_plus
    elif A == '-':
        spm = s_minus
    elif A == 'I':
        spm = s_0

    s = sp.identity(2**i, dtype=complex, format='csr')
    s = sp.kron(s, spm, format='csr')
    s = sp.kron(s, sp.identity(2**(N - i - 1), dtype=complex, format='csr'), format='csr')
    return s


def plising(h, k, lx, ly):
    assert lx >= ly, "Keep the X-direction larger"
    N = lx * ly
    w = np.arange(N)
    x = w % lx
    y = w // lx
    tx = (x + 1) % lx + lx * y
    ty = x + lx * ((y + 1) % ly)
    ty1 = np.concatenate((tx[lx:N], tx[0:lx]), axis=None)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        term = spin("Z", i, N).dot(spin("Z", tx[i], N)).dot(spin("Z", ty[i], N)).dot(spin("Z", ty1[i], N))
        H -= k * term + h * spin("X", i, N)
    return H


def xx_t_zz(h, k, lx, ly):
    assert lx >= ly, "Keep the X-direction larger"
    N = lx * ly
    w = np.arange(N)
    x = w % lx
    y = w // lx
    tx = (x + 1) % lx + lx * y
    ty = x + lx * ((y + 1) % ly)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        H -= k * spin("Z", i, N).dot(spin("Z", tx[i], N)) + h * spin("X", i, N).dot(spin("X", ty[i], N))
    return H


def plising_Sq(h, k, n):
    N = n**2
    w = np.arange(N)
    x = w % n
    y = w // n
    tx = (x + 1) % n + n * y
    ty = x + n * ((y + 1) % n)
    ty1 = np.concatenate((tx[n:N], tx[0:n]), axis=None)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        term = spin("Z", i, N).dot(spin("Z", tx[i], N)).dot(spin("Z", ty[i], N)).dot(spin("Z", ty1[i], N))
        H -= k * term + h * spin("X", i, N)
    return H


def xx_t_zz_Sq(h, k, n):
    N = n**2
    w = np.arange(N)
    x = w % n
    y = w // n
    tx = (x + 1) % n + n * y
    ty = x + n * ((y + 1) % n)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        H -= k * spin("Z", i, N).dot(spin("Z", tx[i], N)) + h * spin("X", i, N).dot(spin("X", ty[i], N))
    return H


def h_tfim(h, J, N, h_v=None):
    if h_v is None:
        h_v = np.zeros(N)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        H -= J * spin("Z", i, N).dot(spin("Z", (i + 1) % N, N))
        H += h * spin("X", i, N) + h_v[i] * spin("Z", i, N)
    return H


def h_xxz(h, J, N, d=0, h_v=None):
    if h_v is None:
        h_v = np.zeros(N)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        H -= J * (spin("X", i, N).dot(spin("X", (i + 1) % N, N)) +
                  spin("Y", i, N).dot(spin("Y", (i + 1) % N, N)) +
                  d * spin("Z", i, N).dot(spin("Z", (i + 1) % N, N)))
        H += h * spin("Z", i, N) + h_v[i] * spin("Z", i, N)
    return H


def h_xxz_obc(h, J, N, d=0, h_v=None):
    if h_v is None:
        h_v = np.zeros(N)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        H -= J * (spin("+", i, N).dot(spin("-", i + 1, N)) +
                  spin("-", i, N).dot(spin("+", i + 1, N)) +
                  d * spin("Z", i, N).dot(spin("Z", i + 1, N)))
        H += h * spin("Z", i, N) + h_v[i] * spin("Z", i, N)
    H += h_v[N - 1] * spin("Z", N - 1, N)
    return H


def h_annni(h, J, J_prime, N, h_v=None):
    if h_v is None:
        h_v = np.zeros(N)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        H -= J * spin("Z", i, N).dot(spin("Z", (i + 1) % N, N))
        H += h * spin("X", i, N) + h_v[i] * spin("Z", i, N)
        H += J_prime * spin("Z", i, N).dot(spin("Z", (i + 2) % N, N))
    return H


def h_annni_obc(h, J, J_prime, N, h_v=None):
    if h_v is None:
        h_v = np.zeros(N)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N - 2):
        H -= J * spin("Z", i, N).dot(spin("Z", i + 1, N))
        H += h * spin("X", i, N) + h_v[i] * spin("Z", i, N)
        H += J_prime * spin("Z", i, N).dot(spin("Z", i + 2, N))
    H -= J * spin("Z", N - 2, N).dot(spin("Z", N - 1, N))
    H += h * spin("X", N - 2, N) + h_v[N - 2] * spin("Z", N - 2, N)
    H += h * spin("X", N - 1, N) + h_v[N - 1] * spin("Z", N - 1, N)
    return H


def h_rfh(J, N, h_v):
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        H += J * (spin("X", i, N).dot(spin("X", (i + 1) % N, N)) +
                  spin("Y", i, N).dot(spin("Y", (i + 1) % N, N)) +
                  spin("Z", i, N).dot(spin("Z", (i + 1) % N, N)))
        H += h_v[i] * spin("Z", i, N)
    return H


def h_rfh_obc(J, N, h_v=None):
    if h_v is None:
        h_v = np.zeros(N)
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        H += J * (spin("X", i, N).dot(spin("X", i + 1, N)) +
                  spin("Y", i, N).dot(spin("Y", i + 1, N)) +
                  spin("Z", i, N).dot(spin("Z", i + 1, N)))
        H += h_v[i] * spin("Z", i, N)
    H += h_v[N - 1] * spin("Z", N - 1, N)
    return H


def h_rfxxz(J, delta, N, h_v):
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        H += J * (spin("X", i, N).dot(spin("X", (i + 1) % N, N)) +
                  spin("Y", i, N).dot(spin("Y", (i + 1) % N, N)))
        H += delta * spin("Z", i, N).dot(spin("Z", (i + 1) % N, N))
        H += h_v[i] * spin("Z", i, N)
    return H


def h_rfxxz_obc(J, delta, N, h_v):
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        H += J * (spin("X", i, N).dot(spin("X", i + 1, N)) +
                  spin("Y", i, N).dot(spin("Y", i + 1, N)))
        H += delta * spin("Z", i, N).dot(spin("Z", i + 1, N))
        H += h_v[i] * spin("Z", i, N)
    H += h_v[N - 1] * spin("Z", N - 1, N)
    return H


def sl2(N):
    s_l2 = sp.csr_matrix((2**N, 2**N), dtype=complex)
    assert N % 2 == 0
    for i in range(N // 2):
        s_l2 += spin("Z", i, N)
    return s_l2


def magnetization(N):
    mz = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        mz += (1.0 / N) * spin("Z", i, N)
    return mz


def entropy(psi_t, r):
    m = np.reshape(psi_t, (r, r))
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    s1 = s**2
    S = -np.dot(s1, np.log(s1))
    x = check_real(S)
    return x


def bipartite_fl(s_l2, psi_t, psi_t_dagg):
    psi_p = s_l2.dot(psi_t)
    psi_p_dagg = np.conjugate(np.transpose(psi_p))
    f = psi_p_dagg.dot(psi_p) - (psi_t_dagg.dot(psi_p))**2
    x = check_real(f)
    return x


def spin_corr(N):
    sc = sp.csr_matrix((2**N, 2**N), dtype=complex)
    for i in range(N):
        sc += spin("Z", i, N).dot(spin("Z", (i + 2) % N, N))
    return sc


def fidelity(psi_0, psi_t, N):
    l = np.log(np.abs(np.vdot(psi_0, psi_t))**2) / N
    x = check_real(l)
    return x


def hermitian(x):
    y = x.getH()
    if (x - y).nnz == 0:
        print("Hermitian")
    else:
        print("Not Hermitian")


def exp_m(e, v, v_dagg, t):
    exp_diag = sp.diags(np.exp(-1j * e * t))
    exp = v.dot(exp_diag).dot(v_dagg)
    return exp


def psi_at_t(e, v, v_dagg, t, psi_0):
    exp_diag = sp.diags(np.exp(-1j * e * t))
    psi_t = v.dot(exp_diag).dot(v_dagg).dot(psi_0)
    return psi_t


def psi_0_xxz(N):
    s = up
    for ii in range(1, N):
        s = np.kron(s, up if ii % 2 == 0 else down)
    return s


def rnd_egnst(sample, N):
    st = np.zeros((sample, 2**N), dtype=complex)
    for jj in range(sample):
        s = 1
        r = np.random.randint(2, size=N)
        for ii in range(N):
            s = np.kron(s, up if r[ii] == 1 else down)
        st[jj] = s
    return st


def check_real(x):
    temp = x
    assert np.allclose(temp.imag, 0), "Imaginary part is not negligible"
    return temp.real
