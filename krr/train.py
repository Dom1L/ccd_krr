import numpy as np
import numba


def learning_curve(subsets, training_features, training_labels, testing_features, testing_labels, train_sigma,
                   train_lambda):
    mae_subsets = []
    for subset in subsets:
        prediction = kernel_training(x_train=training_features[:subset],
                                     y_train=training_labels[:subset],
                                     x_test=testing_features,
                                     train_sigma=train_sigma,
                                     train_lambda=train_lambda)
        mae_subsets.append(np.mean(np.abs(testing_labels - prediction)))
    return mae_subsets


def split_data(x_data, y_data, train_size=0.8, seed=1337):
    train_idx, test_idx = create_split_idx(dataset_size=x_data.shape[0], train_size=train_size, seed=seed)
    training_features = x_data[train_idx]
    training_labels = y_data[train_idx]
    testing_features = x_data[test_idx]
    testing_labels = y_data[test_idx]

    return training_features, training_labels, testing_features, testing_labels


def create_split_idx(dataset_size, train_size=0.8, seed=1337):
    split_idx = np.arange(dataset_size)
    np.random.seed(seed)
    np.random.shuffle(split_idx)
    train_idx = split_idx[:round(dataset_size * train_size)]
    test_idx = split_idx[round(dataset_size * train_size):]
    return train_idx, test_idx


@numba.njit(fastmath=True)
def laplacian_kernel(a, b, sigma=None):
    a_size = a.shape[0]
    b_size = b.shape[0]
    K = np.ones((a_size, b_size))
    i_ind, j_ind = np.triu_indices(n=a_size, m=b_size, k=1)
    i_lower = np.tril_indices(n=a_size, m=b_size, k=-1)
    l1_norm = np.array(manhattan_norm(a, b, i_ind, j_ind))
    if not sigma:
        inv_sigma = -1 / np.max(l1_norm) / np.log(2)
    else:
        inv_sigma = -1 / sigma
    kernel_values = np.exp(l1_norm * inv_sigma)
    K[i_ind, j_ind] = kernel_values
    K[i_lower] = K.T[i_lower]
    return K


@numba.njit(fastmath=True)
def manhattan_norm(a, b, i_ind, j_ind):
    l1_norm = []
    for i, j in zip(i_ind, j_ind):
        l1_norm.append(np.sum(np.abs(a[i] - b[j])))
    return l1_norm


@numba.njit(fastmath=True)
def cho_solve(kernel, y):
    return np.sum(np.linalg.inv(kernel) * y, axis=1)


@numba.njit(fastmath=True)
def kernel_training(x_train, y_train, x_test, train_sigma, train_lambda):
    train_kernel = laplacian_kernel(x_train, x_train, train_sigma)
    train_kernel[np.diag_indices_from(train_kernel)] += train_lambda
    alpha = cho_solve(train_kernel, y_train)

    # calculate a kernel matrix between test and training data, using the same sigma
    test_kernel = laplacian_kernel(x_test, x_train, train_sigma)

    # Make the predictions
    y_pred = kernel_prediction(test_kernel, alpha)
    return y_pred


@numba.njit(fastmath=True)
def kernel_prediction(kernel, alpha):
    return np.dot(kernel, alpha)


if __name__ == '__main__':
    n = 7000
    test_array = np.random.random((n, 300))
    i_ind, j_ind = np.triu_indices(n=n, k=1)
    import time

    start = time.time()
    l = manhattan_norm(test_array, test_array, i_ind, j_ind)
    end = time.time()
    print(start - end)
