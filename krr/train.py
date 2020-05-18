import numpy as np
from tqdm.auto import tqdm
import numba
import random


def learning_curve(subsets, training_features, training_labels, testing_features, testing_labels, train_sigma,
                   train_lambda):
    mae_subsets = []
    for subset in tqdm(subsets):
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


def kernel_training(x_train, y_train, x_test, train_sigma, train_lambda):
    train_kernel, train_sigma = laplacian_kernel(x_train, x_train, train_sigma)
    train_kernel[np.diag_indices_from(train_kernel)] += train_lambda
    alpha = cho_solve(train_kernel, y_train)

    # calculate a kernel matrix between test and training data, using the same sigma
    test_kernel, _ = laplacian_kernel(x_test, x_train, train_sigma)

    # Make the predictions
    y_pred = kernel_prediction(test_kernel, alpha)
    return y_pred


def rkernel_training(x_train, y_train, x_test, y_test, train_sigma, train_lambda, rndm):
    train_kernel, train_sigma = laplacian_kernel(x_train, x_train, train_sigma)
    train_kernel[np.diag_indices_from(train_kernel)] += train_lambda
    alpha = cho_solve(train_kernel, y_train)

    # calculate a kernel matrix between test and training data, using the same sigma
    test_kernel, _ = laplacian_kernel(x_test, x_train, train_sigma)
    new_train_data = rndm#x_train - random.uniform(0, 0.5+n*0.25)
    new_train_kernel, _ = laplacian_kernel(new_train_data, x_train, train_sigma)

    # Make the predictions
    y_trainp = kernel_prediction(train_kernel, alpha)
    print(f'Train MAE: {np.mean(np.abs(y_train-y_trainp))}')
    y_pred = kernel_prediction(test_kernel, alpha)
    print(f'Test MAE: {np.mean(np.abs(y_test-y_pred))}')

    y_train_dp = kernel_prediction(new_train_kernel, alpha)

    return y_pred, y_train_dp, new_train_data


@numba.njit(fastmath=True)
def laplacian_kernel(a, b, sigma=None):
    l1_norm = manhattan_norm(a, b)
    if sigma is None:
        sigma = np.max(l1_norm) / np.log(2)
    return np.exp(-1 * (l1_norm / sigma)),  sigma


@numba.njit(fastmath=True)
def manhattan_norm(a, b):
    a_size = a.shape[0]
    b_size = b.shape[0]
    l1_norm = np.ones((a_size, b_size))
    for i in range(a_size):
        for j in range(b_size):
            l1_norm[i, j] = np.abs(a[i] - b[j])
    return l1_norm


@numba.njit(fastmath=True)
def cho_solve(kernel, y):
    return np.sum(np.linalg.inv(kernel) * y, axis=1)


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
