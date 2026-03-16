# Метод Эйлера
import numpy as np
import matplotlib.pyplot as plt

# число точек на интервале
a = 1
b = 3
hx = 0.1
n = int((b - a) / hx) + 1


def f(x_d, y_d):
    return x_d ** 3 - 2 * y_d


ye = np.zeros(n)
x = np.zeros(n)
x[0] = 0
for i in range(1, n):
    x[i] = x[i - 1] + hx

# Пример dy/dx=f(x,y)
ye[0] = 0
for i in range(0, n - 1):
    ye[i + 1] = ye[i] + hx * f(x[i], ye[i])

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(x, ye,"r*")
print(x)
print(ye)
print(list(range(1, n)))
plt.show()
