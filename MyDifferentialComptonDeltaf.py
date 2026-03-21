import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import math
import time
from tqdm import tqdm
import os
# дискретизация по всем переменным


class KineticEqSolver:
    def __init__(self, Nw, Nx, w_min, w_max, x_min, x_max, beta, T):
        # сетка
        self.T = T
        self.beta = beta
        self.w_grid = np.linspace(w_min, w_max, Nw)
        self.x_grid = np.linspace(x_min, x_max, Nx)
        self.dw = self.w_grid[1] - self.w_grid[0]
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.Nw = Nw
        self.Nx = Nx
        # веса для квадратур
        self.weights_x = np.ones(Nx)
        self.weights_x[0] = self.weights_x[-1] = 0.5
        alpha = 7.297352533 * 10 ** (-3)
        # концентрация
        self.n_e = beta / (2 * np.pi) ** 2 * quad(lambda pz: 1 / math.exp(1 / T * math.sqrt(1 + pz * pz) + 1), -2, 2)[0]
        # коэффициент rho
        self.rho = self.n_e / (2 * np.pi) * alpha ** 2 / (4 ** 2 * np.pi ** 2)
        # коэффициент при ширине
        coefEnGn = 4 * np.pi * alpha * beta ** 2 / (np.pi * math.sqrt(2 * beta + 1) * (math.sqrt(2 * beta + 1) - 1))
        # интеграл в ширине
        integrEnGn = \
            quad(lambda x: math.exp(-x) * (1 - x * (beta + 1) / beta) / math.sqrt(
                x ** 2 - x * (2 * beta + 2) / beta + 1),
                 0,
                 (math.sqrt(2 * beta + 1) - 1) ** 2 / (2 * beta))[0]
        self.EnGn = coefEnGn * math.fabs(integrEnGn)

    def compute_omega_derivatives_from_grid(self, f_slice):
        Nw = len(f_slice)
        df_dw = np.zeros(Nw)
        d2f_dw2 = np.zeros(Nw)

        # Внутренние точки (центральные разности 2-го порядка)
        df_dw[1:-1] = (f_slice[2:] - f_slice[:-2]) / (2 * self.dw)
        d2f_dw2[1:-1] = (f_slice[2:] - 2 * f_slice[1:-1] + f_slice[:-2]) / (self.dw ** 2)

        # Граничные условия (односторонние разности)
        # Левая граница
        df_dw[0] = (-3 * f_slice[0] + 4 * f_slice[1] - f_slice[2]) / (2 * self.dw)
        d2f_dw2[0] = (2 * f_slice[0] - 5 * f_slice[1] + 4 * f_slice[2] - f_slice[3]) / (self.dw ** 2)

        # Правая граница
        df_dw[-1] = (3 * f_slice[-1] - 4 * f_slice[-2] + f_slice[-3]) / (2 * self.dw)
        d2f_dw2[-1] = (2 * f_slice[-1] - 5 * f_slice[-2] + 4 * f_slice[-3] - f_slice[-4]) / (self.dw ** 2)

        return df_dw, d2f_dw2

    def phi_func(self, w_idx, lambdai, lambda_prime, x, xp):
        beta = self.beta
        rho = self.rho
        wtek = self.w_grid[w_idx]
        DeltaM = (wtek ** 2 * (1 - x ** 2) + 2 * wtek - 2 * self.beta) ** 2 + self.EnGn ** 2 / 4
        if lambdai == 0 and lambda_prime == 0:
            return rho * beta ** 2 / DeltaM
        if lambdai == 0 and lambda_prime == 1:
            return rho * wtek ** 2 * xp ** 2 / DeltaM
        if lambdai == 1 and lambda_prime == 1:
            return rho * wtek ** 4 / beta**2 * x ** 2 * xp ** 2 / DeltaM
        if lambdai == 1 and lambda_prime == 0:
            return rho * wtek ** 2 * x ** 2 / DeltaM

    # вычисление правой части

    def compute_rhs_for_point(self, xi, f, lambda_idx, w_idx, x_idx):
        x = self.x_grid[x_idx]
        beta = self.beta
        Delta_omega = self.w_grid[w_idx] - beta
        result = 0
        T = self.T
        for lambda_prime in [0, 1]:
            for xp_idx in range(self.Nx):
                xp = self.x_grid[xp_idx]

                f_lambda_xp = f[lambda_prime, :, xp_idx]

                df_dw, d2f_dw2 = self.compute_omega_derivatives_from_grid(f_lambda_xp)

                kernel = self.phi_func(w_idx, lambda_idx, lambda_prime, x, xp)
                term1 = f_lambda_xp[w_idx] - f[lambda_idx, w_idx, x_idx]
                term2 = (T * df_dw[w_idx] + f_lambda_xp[w_idx]) * (Delta_omega / T)
                term3 = 0.5 * (T ** 2 * d2f_dw2[w_idx] +
                               2 * T * df_dw[w_idx] +
                               f_lambda_xp[w_idx]) * (Delta_omega / T) ** 2

                integrand = kernel * (term1 - term2 + term3)

                weight = self.weights_x[xp_idx] * self.dx
                result += integrand * weight

                # if abs(x) < 1e-10:
                #     result = 0
                # else:
                #     result = x
                return result

        # Формирование ОДУ

    def rhs_system(self, xi, f_flat, pbar, state):
        # Восстановление 3D массива из плоского вектора
        f = f_flat.reshape(2, self.Nw, self.Nx)
        df_dxi = np.zeros_like(f)

        # для трекинга
        last_t, dt = state
        time.sleep(0.1)
        n = int((xi-last_t)/dt)
        pbar.update(n)
        state[0] = last_t + dt * n

        # Для каждой точки вычисляем производную по ξ
        for lambda_idx in range(2):
            for w_idx in range(self.Nw):
                for x_idx in range(self.Nx):
                    df_dxi[lambda_idx, w_idx, x_idx] = 20 * 10 ** 81 * self.compute_rhs_for_point(
                        xi, f, lambda_idx, w_idx, x_idx
                    )
        return df_dxi.flatten()


# Начальные условия
def initial_condition(lambda_idx, w, x):
    A = 1
    ap = 2
    Efold = 40 * 10 ** -3 / 0.5
    return A * w ** -ap * np.exp(-w / Efold)


# Дискретизация по xi
Nxi = 50
xi_max = 100
xi = 5
x = np.cos(np.pi/4)

xi_grid = np.linspace(0, xi_max, Nxi)

# текущий номер z
xitek = np.argmin(xi_grid - xi)

# Собираем правую часть/
solver = KineticEqSolver(Nw=100, Nx=50,
                         w_min=10 / 500, w_max=100 / 500,
                         x_min=-1, x_max=1, beta=0.05, T=0.003 / 0.5)

# текущий номер x
xtek = np.argmin(np.abs(solver.x_grid - x))

# собираем начальные условия
f0 = np.zeros((2, solver.Nw, solver.Nx))

for lambda_idx in range(2):
    for w_idx, w in enumerate(solver.w_grid):
        for x_idx, x in enumerate(solver.x_grid):
            f0[lambda_idx, w_idx, x_idx] = initial_condition(lambda_idx, w, x)

# for lambda_idx in [0,1]:
#     for w_idx in range(solver.Nw):
#         for x_idx in range(solver.Nx):
#             a = solver.compute_rhs_for_point(
#                 xi, f0, lambda_idx, w_idx, x_idx
#             )

if __name__ == "__main__":
    # Решение уравнения
    with tqdm(total = 1000, unit = "‰") as pbar:
        solution = solve_ivp(
            lambda z, f, pbar, state: solver.rhs_system(z, f, pbar, state),
            [0, xi_max],
            f0.flatten(),
            method='RK23',
            t_eval=xi_grid,
            args=[pbar, [0, (xi_max-0)/1000]]
        )

    print(f"     Успех: {solution.success}")
    print(f"     Точек получено: {len(solution.t)}")
    f_solution = solution.y.reshape(2, solver.Nw, solver.Nx, Nxi)

    outfile = os.path.join(os.getcwd() + "\\DataDistF.npz")
    print(outfile)

    # npzfile = np.load(outfile)
    # f_solution = npzfile
    np.savez(outfile, f_solution)

    ax = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    omega = solver.w_grid * 500

    x_label = ['20', '50', '100']

    for lambda_idx, label in enumerate(['λ=1', 'λ=2']):
        plt.subplot(1, 2, lambda_idx + 1)
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
