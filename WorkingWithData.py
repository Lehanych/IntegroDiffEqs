# from MyDifferentialComptonDeltaf import solver, initial_condition, xi_grid
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import math
import time
from tqdm import tqdm
import os
import csv


outfile = os.path.join(os.getcwd(), "DataDistFdw0.045Nxi15ximax100dx0.2.npz")
loadedfile = np.load(outfile)
f_solution = loadedfile["f_solution"]

x_grid = loadedfile["x_grid"]
xi_grid = loadedfile["xi_grid"]
w_grid = loadedfile["w_grid"]


# Фиксируем z
xi = 100

# Фиксируем x
x = np.cos(math.pi/2)

# Вычислени позиции текущего z и x
xtek = np.argmin(np.abs(x_grid - x))
xitek = np.argmin(np.abs(xi_grid - xi))

omega = w_grid*500

xitektab = list((np.argmin(np.abs(xi_grid - xi_i)) for xi_i in [0,50,100]))
print(xitektab)
xtektab = list((np.argmin(np.abs(x_grid - x_i)) for x_i in [0,0.5,1]))
x_label = ['20', '50', '100']

SchSpectra0DegD1 = []
with open('SchSpectra0DegD1.csv', newline='') as csvfile:
    Schreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in Schreader:
        SchSpectra0DegD1.append(row)
    SchSpectra0DegD1 = [list(row) for row in zip(*SchSpectra0DegD1)]

SchSpectra05DegD1 = []
with open('SchSpectra05DegD1.csv', newline='') as csvfile:
    Schreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in Schreader:
        SchSpectra05DegD1.append(row)
    SchSpectra05DegD1 = [list(row) for row in zip(*SchSpectra05DegD1)]

SchSpectra1DegD1 = []
with open('SchSpectra1DegD1.csv', newline='') as csvfile:
    Schreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for row in Schreader:
        SchSpectra1DegD1.append(row)
    SchSpectra1DegD1 = [list(row) for row in zip(*SchSpectra1DegD1)]

for xitek in xitektab:
    i = 0
    for xtek in xtektab:
        ax = plt.figure(figsize=(5, 5))
        match i:
            case 0:
                plt.plot(SchSpectra0DegD1[0], SchSpectra0DegD1[1], linestyle='dashed')
            case 1:
                plt.plot(SchSpectra05DegD1[0], SchSpectra05DegD1[1], linestyle='dashed')
            case 2:
                plt.plot(SchSpectra1DegD1[0], SchSpectra1DegD1[1], linestyle='dashed')
        i += 1

        for lambda_idx, label in enumerate(['λ=1', 'λ=2']):
            plt.xlabel('w, кэВ')
            plt.ylabel('f(w)')
            plt.yscale('log')
            plt.xscale('log')
           # plt.xlim(9, 10 ** 2)
            plt.ylim(1,10**4)
            redfun = f_solution[lambda_idx, :, xtek, xitek]
            plt.plot(omega, redfun, label=label)
        plt.legend()
        plt.title('Функция распределения для x=%s и z=%s' % (round(x_grid[xtek], 2),round(xi_grid[xitek], 2)))
plt.show()


