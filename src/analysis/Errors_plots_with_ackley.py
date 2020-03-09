import numpy as np
import pandas as pd
import seaborn as sns
import estimagic.differentiation.finite_differences as fd
from multiprocessing import Pool
from matplotlib import pyplot as plt
from estimagic.differentiation.generate_steps import generate_steps
from estimagic.optimization.utilities import namedtuple_from_kwargs
from estimagic.differentiation.numdiff_np import first_derivative
from rastrigin import grad_rastrigin
from rastrigin import rastrigin
from levy import func_levy
from levy import grad_levy
from ackley import ackley
from ackley import grad_ackley


def calculate_error(params, diff, func, grad):
    """Difference between numerical and analytical derivatives, devided by
        analytial derivative.

    Args:
        params(np.array): 1d numpy array of function arguments
        diff(np.array): difficulty parameter, controls wiggliness of the function
        func(np.array): functions for which derivatives are calculated
        grad(np.array): gradients of the functions

    Returns:
        error(np.array): numpy array of relative errors, calculated for different
            methods

    """
    method = ["center", "forward", "backward"]
    error = {}
    for i, m in enumerate(method):
        diff_dict = {"diff": diff}
        num_der = first_derivative(func, params, func_kwargs=diff_dict, method=m)
        analytical_der = grad(params, diff)
        error[m] = (num_der - analytical_der) / np.abs(analytical_der).clip(
            1e-8, np.inf
        )

    return error


diff_vec = np.linspace(0, 40, 100)

functions = [rastrigin, func_levy]
gradients = [grad_rastrigin, grad_levy]
function_names = ["Rastrigin", "Levy"]
for f, g, func_name in zip(functions, gradients, function_names):

    error_vec = [
        calculate_error(np.array([5.0, 8.5]), diff_i, f, g) for diff_i in diff_vec
    ]

    to_concat = []
    for err, d in zip(error_vec, diff_vec):
        df = pd.DataFrame(err)
        df["difficulty"] = d
        to_concat.append(df)

    data = pd.concat(to_concat)

    data["param_name"] = "x_" + (data.index + 1).astype(str)

    params = data.param_name.unique().tolist()
    methods = ["center", "forward", "backward"]
    fig, axes = plt.subplots(nrows=len(params), figsize=(5, 3 * len(params)))

    for param, ax in zip(params, axes.flatten()):
        df = data.query(f"param_name == '{param}'")
        for method in methods:
            sns.lineplot(ax=ax, y=df[method], x=df["difficulty"], label=method)
        ax.set(ylabel=r"$\frac{\partial f}{\partial " + param + "}$")
    fig.suptitle("Error of " + func_name + " function Gradient", y=1)
    plt.tight_layout()
    fig.savefig(func_name + ".png")

# plot errors ackley

for i in range(3):
    diff_i = np.array([20, 0.2, 2 * np.pi])
    error_vec = [
        calculate_error(np.array([5.0, 8.5]), diff_i, ackley, grad_ackley)
        for diff_i[i] in diff_vec
    ]

    to_concat = []
    for err, d in zip(error_vec, diff_vec):
        for k in err.keys():
            if len(err[k]) == 1:
                print(k, d)
        df = pd.DataFrame(err)

        df["difficulty"] = d
        to_concat.append(df)

    data = pd.concat(to_concat)

    data["param_name"] = "x_" + (data.index + 1).astype(str)

    params = data.param_name.unique().tolist()
    methods = ["center", "forward", "backward"]
    fig, axes = plt.subplots(nrows=len(params), figsize=(5, 3 * len(params)))

    for param, ax in zip(params, axes.flatten()):
        df = data.query(f"param_name == '{param}'")
        for method in methods:
            sns.lineplot(ax=ax, y=df[method], x=df["difficulty"], label=method)
        ax.set(
            ylabel=r"$\frac{\partial f}{\partial " + param + "}$",
            xlabel="difficulty" + str(i),
        )

    plt.tight_layout()
    fig.savefig("Ackley" + str(i) + ".png")
