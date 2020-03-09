import numpy as np
import pandas as pd
import seaborn as sns
import estimagic.differentiation.finite_differences as fd
from multiprocessing import Pool
from matplotlib import pyplot as plt
from estimagic.differentiation.generate_steps import generate_steps
from estimagic.optimization.utilities import namedtuple_from_kwargs
from estimagic.differentiation.numdiff_np import first_derivative
from src.model_code.rastrigin import grad_rastrigin
from src.model_code.rastrigin import rastrigin
from src.model_code.levy import func_levy
from src.model_code.levy import grad_levy
from src.model_code.ackley import ackley
from src.model_code.ackley import grad_ackley
from scipy.optimize._numdiff import approx_derivative
from bld.project_paths import project_paths_join as ppj



if __name__ == "__main__":
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
        for i,m in enumerate(method):
            diff_dict = {'diff': diff}
            num_der = first_derivative(func, params,func_kwargs=diff_dict,
                          method=m)
            analytical_der = grad(params, diff)
            error[m] = (num_der - analytical_der )/np.abs(analytical_der).clip(1e-8, np.inf)

        return error

    diff_vec = np.linspace(0, 40, 100)

    functions = [rastrigin, func_levy]
    gradients = [grad_rastrigin, grad_levy]
    function_names = ["Rastrigin", "Levy"]
    for f,g, func_name in zip(functions, gradients, function_names):

        error_vec = [calculate_error(np.array([5.0, 8.5]),diff_i, f, g) for
            diff_i in diff_vec]

        to_concat = []
        for err, d in zip(error_vec, diff_vec):
            df = pd.DataFrame(err)
            df["difficulty"] = d
            to_concat.append(df)

        data = pd.concat(to_concat)

        data["param_name"] = "x_" + (data.index+1).astype(str)


        params= data.param_name.unique().tolist()
        methods = ["center", "forward", "backward"]
        fig, axes = plt.subplots(nrows=len(params), figsize=(5,  3* len(params)))

        for param, ax in zip(params, axes.flatten()):
            df = data.query(f"param_name == '{param}'")
            for method in methods:
                sns.lineplot(ax=ax, y=df[method], x=df["difficulty"], label=method)
            ax.set(ylabel=r'$\frac{\partial f}{\partial '+param+'}$')
        fig.suptitle('Error of ' +func_name+' function Gradient', y=1)
        plt.tight_layout()
        with open(ppj("OUT_FIGURES", func_name + ".png"), "wb") as j:
            fig.savefig(j)
    # plot errors ackley
