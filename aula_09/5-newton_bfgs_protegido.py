import numpy as np
import time
from scipy.optimize import minimize
from sympy import symbols, diff, hessian, lambdify, exp, log

# ==========================================
# 1. SETUP SIMBÓLICO COM PROTEÇÃO NUMÉRICA
# ==========================================
x_s, y_s = symbols("x y")

# Função de custo da GreenLog
f_sym = (
    exp(x_s - 2) + exp(y_s - 1) - log(1 + x_s**2 + y_s**2) + (x_s - 2 * y_s) ** 2 + 10
)

# Gradiente e Hessiana automáticos
grad_sym = [diff(f_sym, var) for var in (x_s, y_s)]
hess_sym = hessian(f_sym, (x_s, y_s))

# DICIONÁRIO DE PROTEÇÃO: Substituímos o exp padrão por um 'exp_safe'
# Isso limita o expoente entre -700 e 700 para evitar o RuntimeWarning
safe_module = {
    "exp": lambda z: np.exp(np.clip(z, -700, 700)),
    "log": lambda z: np.log(np.maximum(z, 1e-15)),  # Proteção extra para log(0)
}

# "Compilando" as funções com o módulo de segurança
f_num = lambdify((x_s, y_s), f_sym, modules=[safe_module, "numpy"])
grad_num = lambdify((x_s, y_s), grad_sym, modules=[safe_module, "numpy"])
hess_num = lambdify((x_s, y_s), hess_sym, modules=[safe_module, "numpy"])


# Helpers para compatibilidade com SciPy
def f_scipy(x_vec):
    return f_num(x_vec[0], x_vec[1])


def g_scipy(x_vec):
    return np.array(grad_num(x_vec[0], x_vec[1]))


# ==========================================
# 2. ALGORITMOS (Newton e BFGS)
# ==========================================


def rodar_newton_manual(p0, tol=1e-6, max_iter=50):
    p = np.array(p0, dtype=float)
    for i in range(max_iter):
        g = np.array(grad_num(p[0], p[1]))
        H = np.array(hess_num(p[0], p[1]))

        if np.linalg.norm(g) < tol:
            return p, i

        try:
            # Resolve H * p = g de forma estável
            p = p - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return p, i
    return p, max_iter


# ==========================================
# 3. EXECUÇÃO DO BENCHMARK
# ==========================================
p_init = [0.0, 0.0]  # Ponto inicial

# Newton (Semana 8)
t0 = time.time()
res_newton, it_newton = rodar_newton_manual(p_init)
t_newton = (time.time() - t0) * 1000

# BFGS (Semana 9)
t0 = time.time()
res_bfgs = minimize(f_scipy, p_init, method="BFGS", jac=g_scipy, tol=1e-6)
t_bfgs = (time.time() - t0) * 1000

# ==========================================
# 4. RESULTADOS
# ==========================================
print("\n" + "=" * 60)
print(f"{'MÉTODO':<20} | {'ITERAÇÕES':<10} | {'TEMPO (ms)':<12} | {'CUSTO'}")
print("-" * 60)
print(
    f"{'Newton (Manual)':<20} | {it_newton:<10} | {t_newton:<12.4f} | {f_scipy(res_newton):.6f}"
)
print(
    f"{'BFGS (SciPy)':<20} | {res_bfgs.nit:<10} | {t_bfgs:<12.4f} | {res_bfgs.fun:.6f}"
)
print("=" * 60)
