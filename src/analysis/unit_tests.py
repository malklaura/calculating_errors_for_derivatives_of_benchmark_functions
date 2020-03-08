""" Tests for functions, gradients and hessians."""
import sys
import os
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from src.model_code.ackley import ackley
from src.model_code.ackley import grad_ackley
from src.model_code.ackley import hessian_ackley
from src.model_code.levy import func_levy
from src.model_code.levy import grad_levy
from src.model_code.levy import hessian_levy
from src.model_code.rastrigin import rastrigin
from src.model_code.rastrigin import grad_rastrigin
from src.model_code.rastrigin import hessian_rastrigin

"""Unit tests for Ackley function, gradient and hessian."""
@pytest.fixture
def set_up_ackley():
    out = {}
    out["params"] = np.array([1.0, 1.0])
    out["diff"] = np.array([0.2, 20, 2 * np.pi])
    return out


def test_function_ackley(set_up_ackley):
    exp = 0.2
    res = ackley(**set_up_ackley)
    aaae(exp, res)


def test_function_ackley_grad(set_up_ackley):
    exp = np.array([[0.0, 0.0]])
    res = grad_ackley(**set_up_ackley)
    aaae(exp, res)


def test_function_ackley_hess(set_up_ackley):
    exp = np.array([[5.36567326e01, 4.32842261e-8], [4.32842261e-8, 5.36567326e01]])
    res = hessian_ackley(**set_up_ackley)
    aaae(exp, res)


"""Unit tests for Levy function, gradient and hessian."""
@pytest.fixture
def set_up_levy():
    out = {}
    out["params"] = np.array([1.0, 3.0])
    out["diff"] = 3.0 * np.pi
    return out


def test_function_levy(set_up_levy):
    exp = 4.0
    res = func_levy(**set_up_levy)
    aaae(exp, res)


def test_function_levy_grad(set_up_levy):
    exp = np.array([0.0, 4.0])
    res = grad_levy(**set_up_levy)
    aaae(exp, res)


def test_function_levy_hess(set_up_levy):
    exp = np.array([[179.65287922, 0.0], [0.0, 317.82734083]])
    res = hessian_levy(**set_up_levy)
    aaae(exp, res)



"""Unit tests for Rastrigin function, gradient and hessian."""
@pytest.fixture
def set_up_rast():
    out = {}
    out["params"] = np.array([5.0, 8.0])
    out["diff"] = 2.0 * np.pi
    return out


def test_function_rast(set_up_rast):
    exp = 89.0
    res = rastrigin(**set_up_rast)
    aaae(exp, res)


def test_function_rast_grad(set_up_rast):
    exp = np.array([10.0, 16.0])
    res = grad_rastrigin(**set_up_rast)
    aaae(exp, res)


def test_function_rast_hess(set_up_rast):
    exp = np.array([[396.784176044, 0.0], [0.0, 396.784176044]])
    res = hessian_rastrigin(**set_up_rast)
    aaae(exp, res)

if __name__ == "__main__":
    status = pytest.main([sys.argv[1]])
    sys.exit(status)
