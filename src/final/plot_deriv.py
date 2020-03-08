import numpy as np
from matplotlib import pyplot as plt
from src.model_code.ackley import grad_ackley
from src.model_code.rastrigin import grad_rastrigin
from src.model_code.levy import grad_levy

"""Plot gardient of Ackley function for params = [1, 2]."""
params = np.array([1, 2])
a = np.linspace(0, 50, 100)
b = np.linspace(0, 10, 100)
c = np.linspace(0, 20, 100)

plt_a = np.array([grad_ackley(params, np.array([ai, 0.2, 2 * np.pi])) for ai in a])
plt_b = np.array([grad_ackley(params, np.array([20, bi, 2 * np.pi])) for bi in b])
plt_c = np.array([grad_ackley(params, np.array([20, 0.2, ci])) for ci in c])
figure, axes = plt.subplots(nrows=3, ncols=1)

plt.xlabel("difficulty")
axes[0].plot(a, plt_a)
axes[1].plot(b, plt_b)
axes[2].plot(c, plt_c)
axes[0].set_xlabel("a")
axes[1].set_xlabel("b")
axes[2].set_xlabel("c")
figure.suptitle("Gradient of Ackley function")
figure.tight_layout()

"""Plot gardient of Rastrigin function for params = [1, 2, 5]."""
params = np.array([1, 2, 5])
diff = np.linspace(0, 20, 100)
plt_ras = np.array([grad_rastrigin(params, diff_i) for diff_i in diff])

figure, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
figure.suptitle("Gradient of Ackley function")
plt.xlabel("difficulty")
axes[0].plot(diff, plt_ras[:, 0])
axes[1].plot(diff, plt_ras[:, 1])
axes[2].plot(diff, plt_ras[:, 2])
figure.tight_layout()

"""Plot gardient of Levy function for params = [1.5, 5.5]."""
params = np.array([1.5, 2.5])
diff = np.linspace(0, 20, 100)
plt_lev = np.array([grad_levy(params, diff_i) for diff_i in diff])
figure, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
figure.suptitle("Gradient of Levy function")
plt.xlabel("difficulty")
axes[0].plot(diff, plt_lev[:, 0])
axes[1].plot(diff, plt_lev[:, 1])
figure.tight_layout()
