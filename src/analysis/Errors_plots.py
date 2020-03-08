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



def calculate_error(params, diff, func, grad):
    method = ["center", "forward", "backward"]
    error = {}
    for i,m in enumerate(method):
        diff_dict = {'diff': diff}
        num_der = first_derivative(func, params,func_kwargs=diff_dict,
                      method=m)
        analytical_der = grad(params, diff)
        error[m] = (num_der - analytical_der )/np.abs(analytical_der).clip(1e-8, np.inf)
        #error[m] = first_derivative(func, params,func_kwargs=diff_dict,
        #              method=m) - grad(params, diff)
    return error
calculate_error(np.array([1,2]), np.array([1,2,3]), ackley, grad_ackley)
#k = int(len(diff))
#diff_vec = np.zeros_like(diff)
#for j in range(k):
#    diff_vec[j] = np.linspace(0, 40, 100)
#return np.array([diff_vec])

diff_vec = np.linspace(0, 40, 100)

functions = [rastrigin, func_levy, ackley]
gradients = [grad_rastrigin, grad_levy, grad_ackley]
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

    data["param_name"] = "x_" + data.index.astype(str)


    params= data.param_name.unique().tolist()
    methods = ["center", "forward", "backward"]
    fig, axes = plt.subplots(nrows=len(params), figsize=(5, 3 * len(params)))

    for param, ax in zip(params, axes.flatten()):
        df = data.query(f"param_name == '{param}'")
        for method in methods:
            sns.lineplot(ax=ax, y=df[method], x=df["difficulty"], label=method)



    #for function_name in enumerate(function_names):
    #    fig.savefig(function_name.png)

#def save_fig(data, name, filename):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.hist(data, color="blue")
#    ax.set_title(name)
#    fig.tight_layout()
#    fig.savefig("{}_{}.png".format(filename,name), format='png')
#    plt.close(fig)

#filename = 'output_graph_2'
#for data, name in zip(list_of_data, list_of_names):
#    save_hist(data, name, filename)
