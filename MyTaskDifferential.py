from fenics import *

# Создание сетки
mesh = UnitSquareMesh(32, 32)

# Определение функционального пространства
V = FunctionSpace(mesh, 'P', 1)

# Определение граничных условий
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Определение вариационной задачи
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)  # Правая часть уравнения
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Решение задачи
u = Function(V)
solve(a == L, u, bc)

# Визуализация результата
plot(u)
plt.show()
