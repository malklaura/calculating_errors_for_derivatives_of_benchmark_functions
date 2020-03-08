import numpy as np


def func_levy(params, diff):
    """Levy function N13, which is:
    f(x) = sin^{2}(3\pi x_{1}) + (x_{1}-1)^{2}[1+sin^{2}(c \pi x_2)]
             + (x_2-1)^{2}[1+sin^{2}(2 \pi x_2)]]
    link to the function: https://tinyurl.com/tfn3o7p

    Args:
        params(np.array): 1d numpy array of function arguments
        diff(float): difficulty parameter, controls wiggliness of the function

    Returns:
        func(float): value of the Levy function evaluated at `params` and `diff`

    """
    param1 = params[0]
    param2 = params[1]
    func = (
        (np.sin(3 * np.pi * param1)) ** 2
        + (param1 - 1) ** 2 * (1 + np.sin(diff * param2) ** 2)
        + (param2 - 1) ** 2 * (1 + np.sin(2 * np.pi * param2) ** 2)
    )
    return func


def grad_levy(params, diff):
    """Gradient of Levy functio N13.

    Args:
        params(np.array): 1d numpy array of function arguments
        diff(float): difficulty parameter, controls wiggliness of the function

    Returns:
        grad(np.array): 1d numpy array of Levy function derivative for each
            argument, evaluated at `params` and `diff`

    """
    param1 = params[0]
    param2 = params[1]
    der_param1 = 6 * np.pi * np.cos(3 * np.pi * param1) * np.sin(
        3 * np.pi * param1
    ) + 2 * ((np.sin(diff * param2)) ** 2 + 1) * (param1 - 1)
    der_param2 = (
        2 * diff * (param1 - 1) ** 2 * np.cos(diff * param2) * np.sin(diff * param2)
        + 2 * (param2 - 1) * ((np.sin(2 * np.pi * param2)) ** 2 + 1)
        + 4
        * np.pi
        * (param2 - 1) ** 2
        * np.cos(2 * np.pi * param2)
        * np.sin(2 * np.pi * param2)
    )

    grad = np.array([der_param1, der_param2])
    return grad


def hessian_levy(params, diff):
    """Hessian matrix of Levy function.

    Args:
        params(np.array): 1d array of function arguments
        diff(float): difficulty parameter, controls wiggliness of the function

    Returns:
        hess_mat(np.array): 2d numpy array with the Levy function hessian,
            evaluated at `params` and `diff`

    """
    param1 = params[0]
    param2 = params[1]

    der_param1_param1 = (
        -18 * np.pi ** 2 * (np.sin(3 * np.pi * param1)) ** 2
        + 18 * np.pi ** 2 * (np.cos(3 * np.pi * param1)) ** 2
        + 2 * ((np.sin(diff * param2)) ** 2 + 1)
    )
    der_param1_param2 = (
        4 * diff * (param1 - 1) * np.cos(diff * param2) * np.sin(diff * param2)
    )
    der_param2_param1 = (
        4 * diff * (param1 - 1) * np.cos(diff * param2) * np.sin(diff * param2)
    )
    der_param2_param2 = (
        2 * diff ** 2 * (param1 - 1) ** 2 * np.cos(2 * diff * param2)
        + 8 * np.pi * (param2 - 1) * np.sin(4 * np.pi * param2)
        + (8 * np.pi ** 2 * (param2 - 1) ** 2 - 1) * np.cos(4 * np.pi * param2)
        + 3
    )

    hess_mat =  np.array(
        [[der_param1_param1, der_param1_param2], [der_param2_param1, der_param2_param2]]
    )
    return hess_mat
