import numpy as np


def ackley(params, diff):
    """Ackley function, which is:
    f(x) = -a exp(-b\sqrt{\frac{1}{d} \sum_{i=1}^n x_i^2})
            - exp(\frac{1}{d} \sum_{i=1}^n cos(c x_i)) + a + e
    link to the function: https://tinyurl.com/qmds7dh

    Args:
        params(np.array): 1d numpy array of function arguments
        diff(np.array): 1d np.array of difficulty parameters
            First value(a) defines the 'height' of the figure, larger (a) makes the 'cone'
            in the figure higher. Second value(b) generates a 'hole', the larger (b) is the
            wider  is the hole. Third value(c) defines the wiggliness of the function. The
            higer (c) gets, fluctuations become more frequent, so it gets more difficult to
            take numerical derivative.

    Returns:
           res(float): value of the function evaluated at `params` and `diff`

    """
    a = np.array(diff[0])
    b = np.array(diff[1])
    c = np.array(diff[2])
    sum1 = np.sum(params**2)
    sum2 = np.sum(np.cos(c*params))
    d = int(len(params))
    res = -a*np.exp(-b*np.sqrt(sum1/d)) - np.exp(sum2/d) + a + np.e
    return res

def grad_ackley(params, diff):
    """Gradient vector of Ackley function.

    Args:
        params(np.array): 1d numpy array of function arguments
        diff(np.array): 1d np.array of difficulty parameters

    Returns:
        grad(np.array): 1d numpy array of Ackley function derivative for each
            argument, evaluated at `params` and `diff`
    """

    a = np.atleast_1d(diff[0])
    b = np.atleast_1d(diff[1])
    c = np.atleast_1d(diff[2])
    sum1 = np.sum(params**2)
    sum2 = np.sum(np.cos(c*params))
    d = int(len(params))
    der_param = np.zeros_like(params)
    for i in range(d):
        der_param[i] = a * b / d * params[i] * np.exp(-b*np.sqrt(sum1/d))/np.sqrt(sum1/d)
        + c/d * np.exp(1/d*sum2) * np.sin(c*params[i])

    grad = np.array([der_param])
    return grad

def hessian_ackley(params, diff):
    """Return Hessian matrix of Ackley function."""
    """Hessioan matrix of Ackley function.

    Args:
        params(np.array): 1d numpy array of function arguments
        diff(np.array): 1d np.array of difficulty parameters

    Returns:
        hess_mat(np.array): 2d numpy array with the Ackley function hessian,
            evaluated at `params` and `diff`

    """
    a = diff[0]
    b = diff[1]
    c = diff[2]
    sum1 = np.sum(params**2)
    sum2 = np.sum(np.cos(c*params))
    d = int(len(params))
    hess_mat = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if i == j:
                hess_mat[i,j] = (a * b / d * np.exp(-b*np.sqrt(sum1/d)) * (1/np.sqrt(sum1/d) -
                           b * params[i]**2 / sum1 - params[i]**2/d * (1/d*sum1)**(-3/2)) +
                            c**2/d*np.exp(1/d*sum2)*(np.cos(c*params[i]) - 1/d*(np.sin(c*params[i]))**2))
            else:
                hess_mat[i,j] =( -a*b / d * params[i] * params[j] * np.exp(-b*np.sqrt(sum1/d)) * (b/sum1 + 1/d * (1/d*sum1)**(-3/2)) -
                            c**2/d**2 * np.exp(1/d*sum2) * np.sin(c*params[i]) * np.sin(c*params[j]))

    return hess_mat
