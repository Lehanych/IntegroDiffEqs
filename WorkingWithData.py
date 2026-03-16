from MyDifferentialComptonDeltaf import solver, initial_condition, xi_grid
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import math
import time
from tqdm import tqdm
import os

xi = 100
x = np.cos(math.pi / 2)
xtek = np.argmin(np.abs(solver.x_grid - x))


xitek = np.argmin(np.abs(xi_grid - xi))

outfile = os.getcwd() + "\DataDistF.npz"
f_solution = np.load(outfile)["arr_0"]


ax = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
omega = solver.w_grid * 500

x_label = ['20', '50', '100']

for lambda_idx, label in enumerate(['λ=1', 'λ=2']):
    plt.subplot(1, 2, 1)

    plt.xlabel('w, кэВ')
    plt.ylabel('f(w)')
    plt.yscale('log')
    plt.xscale('log')
    redfun = f_solution[lambda_idx, :, xtek, xitek]
    plt.plot(omega, redfun, label=label)

plt.subplot(1, 2, 2)
plt.legend()
for lambda_idx, label in enumerate(['λ=1', 'λ=2']):
    redfun = initial_condition(lambda_idx, omega / 500, xtek)
    plt.plot(omega, redfun, label=label)

plt.subplot(1, 2, 1)
plt.legend()
plt.title('Функция распределения для x=%s' % round(solver.x_grid[xtek], 2))

plt.show()
