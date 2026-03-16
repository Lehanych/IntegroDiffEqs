import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

N = 50 # Количество узлов в сетке
x = np.linspace(0, 1, N)

k = 0.1
u0 = np.sin(np.pi * x)

diagonals = np.zeros((3, N)) # Создаем матрицу с тремя диагоналями
diagonals[0,:] = -1
diagonals[1,:] = 2
diagonals[2,:] = -1
A = sparse.spdiags(diagonals, [-1,0,1], N, N, format="csr") # Создаем разреженную матрицу

for i in range(100): # 100 временных шагов
    b = k * u0
    u = spsolve(A, b) # Решаем систему уравнений
    plt.plot(x, u) # Визуализируем результат
    u0 = u
plt.show()