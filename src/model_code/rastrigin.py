"""Rastrigin function and derivatives."""
import numpy as np


def rastrigin(params, diff):
    """Rastrigin function, which is:
    f(x) = 10*d + \sum_{i=1}^{d}(x_i^{2}-10cos(cx_i))
    link to the function: https://tinyurl.com/s739unq

    Args:
        params(np.array): 1d numpy array of function arguments
        diff(float): difficulty parameter, controls wiggliness of the function

    Returns:
        func(float): value of the function evaluated at `params` and `diff`

    """

    res = 10 * len(params) + np.sum(params ** 2 - 10 * np.cos(diff * params))
    return res


def grad_rastrigin(params, diff):
    """Gradient of Rastrigin function.

    Args:
        params(np.array): 1d numpy array of function arguments
        diff(float): difficulty parameter, controls wiggliness of the function

    Returns:
        grad(np.array): 1d numpy array of Rastrigin function derivatives for each
        argument, evaluated at `params` and `diff`

    """
    d = int(len(params))
    term_1 = np.zeros_like(params)
    term_2 = np.zeros_like(params)
    grad = np.zeros_like(params)
    for i in range(d):
        term_1[i] = 2 * params[i]
        term_2[i] = 10 * diff * np.sin(diff * params[i])
        grad[i] = term_1[i] + term_2[i]
    return grad


def hessian_rastrigin(params, diff):
    """Hessian matrix of Rastrigin function.

    Args:
        params(np.array): 1d array of function arguments
        diff(float): difficulty parameter, controls wiggliness of the function

    Returns:
        hess_mat(np.array): 2d numpy array with the Rastrigin function hessian,
        evaluated at `params` and `diff`

    """
    d = int(len(params))
    hess_mat = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                hess_mat[i, j] = 2 + 10 * diff ** 2 * np.cos(diff * params[i])
            else:
                hess_mat[i, j] = 0.0

    return hess_mat
