import numpy as np
import matplotlib.pyplot as plt

# Задаем параметры
alpha = 1.0
beta = 0.1
delta = 0.075
gamma = 1.5

# Задаем начальные условия
x = 10
y = 5

# Задаем временной интервал и шаг
t = 0
t_end = 100
dt = 0.01

# Создаем списки для сохранения значений популяции и времени
time = []
x_values = []
y_values = []

while t < t_end:
    # Подсчет инкрементов для x и y
    dx1 = dt * (alpha * x - beta * x * y)
    dy1 = dt * (delta * x * y - gamma * y)

    dx2 = dt * (alpha * (x + dx1 / 2) - beta * (x + dx1 / 2) * (y + dy1 / 2))
    dy2 = dt * (delta * (x + dx1 / 2) * (y + dy1 / 2) - gamma * (y + dy1 / 2))

    dx3 = dt * (alpha * (x + dx2 / 2) - beta * (x + dx2 / 2) * (y + dy2 / 2))
    dy3 = dt * (delta * (x + dx2 / 2) * (y + dy2 / 2) - gamma * (y + dy2 / 2))

    dx4 = dt * (alpha * (x + dx3) - beta * (x + dx3) * (y + dy3))
    dy4 = dt * (delta * (x + dx3) * (y + dy3) - gamma * (y + dy3))

    # Обновляем значения x и y
    x += (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6
    y += (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6

    # Сохраняем текущие значения
    time.append(t)
    x_values.append(x)
    y_values.append(y)

    # Обновляем время
    t += dt

# Рисуем график численности популяций
plt.plot(time, x_values, label='Preys')
plt.plot(time, y_values, label='Predators')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()