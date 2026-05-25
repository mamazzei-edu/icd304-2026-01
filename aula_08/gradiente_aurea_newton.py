from sympy import symbols, hessian, lambdify
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog
import matplotlib.pyplot as plt


def custo(x, y):
    return (x - 2) ** 4 + (x - 2 * y) ** 2 + 10


# Definindo as variáveis simbólicas e a função de custo
x, y = symbols("x y")
# Calculando a função de custo, a Hessiana e as versões lambdificadas
f = custo(x, y)
H = hessian(f, (x, y))
f_lambdified = lambdify((x, y), f)
H_lambdified = lambdify((x, y), H)


# 1. Definições da Função GreenLog
def custo(p):
    return (p[0] - 2) ** 4 + (p[0] - 2 * p[1]) ** 2


def gradiente(p):
    df_dx = 4 * (p[0] - 2) ** 3 + 2 * (p[0] - 2 * p[1])
    df_dy = -4 * (p[0] - 2 * p[1])
    return np.array([df_dx, df_dy])


def hessiana(p):
    d2f_dx2 = 12 * (p[0] - 2) ** 2 + 2
    d2f_dy2 = 8
    d2f_dxdy = -4
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])


# 2. Implementação Manual da Razão Áurea (Semana 6)
def gss_manual(func, a, b, tol=1e-5):
    rho = (np.sqrt(5) - 1) / 2
    d = rho * (b - a)
    x1, x2 = b - d, a + d
    f1, f2 = func(x1), func(x2)
    while (b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - rho * (b - a)
            f1 = func(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + rho * (b - a)
            f2 = func(x2)
    return (a + b) / 2


# 3. COMPETIDORES (OS ALGORITMOS)


def rodar_gradiente_simples(p0, alpha=0.01, max_iter=2000):
    p = np.array(p0)
    for i in range(max_iter):
        g = gradiente(p)
        if np.linalg.norm(g) < 1e-6:
            return p, i
        p = p - alpha * g
    return p, max_iter


def rodar_steepest_manual(p0, max_iter=100):
    p = np.array(p0)
    for i in range(max_iter):
        g = gradiente(p)
        if np.linalg.norm(g) < 1e-6:
            return p, i
        # Busca por Linha usando GSS Manual
        phi = lambda a: custo(p - a * g)
        step = gss_manual(phi, 0, 1)
        p = p - step * g
    return p, max_iter


def rodar_steepest_scipy(p0, max_iter=100):
    p = np.array(p0)
    for i in range(max_iter):
        g = gradiente(p)
        if np.linalg.norm(g) < 1e-6:
            return p, i
        # Busca por Linha usando SciPy (Razão Áurea Industrial)
        phi = lambda a: custo(p - a * g)
        step = minimize_scalar(phi, bracket=(0, 1), method="golden").x
        p = p - step * g
    return p, max_iter


def rodar_newton(p0, max_iter=100):
    p = np.array(p0)
    for i in range(max_iter):
        g = gradiente(p)
        if np.linalg.norm(g) < 1e-6:
            return p, i
        H = hessiana(p)
        p = p - np.linalg.solve(H, g)
    return p, i


# 4. EXECUÇÃO DO BENCHMARK
p_start = [0.0, 3.0]
metodos = [
    ("Gradiente Simples (Alpha Fixo)", lambda: rodar_gradiente_simples(p_start)),
    ("Steepest Descent (GSS Manual)", lambda: rodar_steepest_manual(p_start)),
    ("Steepest Descent (SciPy GSS)", lambda: rodar_steepest_scipy(p_start)),
    ("Método de Newton (Segunda Ordem)", lambda: rodar_newton(p_start)),
]

print(f"{'Algoritmo':<35} | {'Iterações':<10} | {'Tempo (ms)':<10}")
print("-" * 65)

for nome, func in metodos:
    t0 = time.time()
    res, iters = func()
    t1 = (time.time() - t0) * 1000  # milissegundos
    print(f"{nome:<35} | {iters:<10} | {t1:<10.4f}")
