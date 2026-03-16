import sympy as sp

# Определяем переменные
t = sp.symbols('t')
y = sp.Function('y')

# Определяем уравнение
equation = sp.Eq(y(t).diff(t), -2 * y(t))

# Решаем уравнение
solution = sp.dsolve(equation)

print(solution)