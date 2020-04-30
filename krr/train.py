import numpy as np
import numba


def laplacian_kernel(a, b):
    a_size = a.shape[0]
    b_size = b.shape[0]
    K = np.ones((a_size, b_size))
    i_ind, j_ind = np.triu_indices(n=a_size, m=b_size, k=1)
    i_lower = np.tril_indices(n=a_size, m=b_size, k=-1)
    l1_norm = np.array(fast_manhattan(a, b, i_ind, j_ind))
    inv_sigma = -1/np.max(l1_norm)/np.log(2)
    kernel_values = np.exp(l1_norm * inv_sigma)
    K[i_ind, j_ind] = kernel_values
    K[i_lower] = K.T[i_lower]
    return K

@numba.njit(fastmath=True)
def fast_manhattan(a, b, i_ind, j_ind):
    l1_norm = []
    for i, j in zip(i_ind, j_ind):
        l1_norm.append(np.sum(np.abs(a[i] - b[j])))
    return l1_norm


def cho_solve(kernel, y):
    return np.sum(np.linalg.inv(kernel) * y, axis=1)


def train_krr():
    pass


if __name__ == '__main__':
    n=7000
    test_array = np.random.random((n, 300))
    i_ind, j_ind = np.triu_indices(n=n, k=1)
    l = fast_manhattan(test_array, test_array, i_ind, j_ind)
