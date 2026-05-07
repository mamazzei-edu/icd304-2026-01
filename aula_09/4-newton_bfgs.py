import numpy as np
import time
from scipy.optimize import minimize
from sympy import symbols, diff, hessian, lambdify, exp, log

# ==========================================
# 1. SETUP SIMBÓLICO (Para o Newton "Braçal")
# ==========================================
x_s, y_s = symbols('x y')
f_sym = exp(x_s - 2) + exp(y_s - 1) - log(1 + x_s**2 + y_s**2) + (x_s - 2*y_s)**2

# Para o Newton, PRECISAMOS da Hessiana
grad_sym = [diff(f_sym, var) for var in (x_s, y_s)]
hess_sym = hessian(f_sym, (x_s, y_s))

# "Compilando" para funções rápidas
f_num    = lambdify((x_s, y_s), f_sym, 'numpy')
grad_num = lambdify((x_s, y_s), grad_sym, 'numpy')
hess_num = lambdify((x_s, y_s), hess_sym, 'numpy')

# Helper para o SciPy (aceita vetor x)
def f_scipy(x_vec):
    return f_num(x_vec[0], x_vec[1])

def g_scipy(x_vec):
    return np.array(grad_num(x_vec[0], x_vec[1]))

# ==========================================
# 2. MÉTODO DE NEWTON (O que vimos na Semana 8)
# ==========================================
def rodar_newton_manual(p0, tol=1e-6, max_iter=50):
    p = np.array(p0, dtype=float)
    for i in range(max_iter):
        g = np.array(grad_num(p[0], p[1]))
        H = np.array(hess_num(p[0], p[1]))
        
        if np.linalg.norm(g) < tol:
            return p, i
            
        try:
            p = p - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return p, i
    return p, max_iter

# ==========================================
# 3. BENCHMARK E COMPARAÇÃO
# ==========================================
p_init = [0.0, 0.0]

print("--- INICIANDO COMPARAÇÃO: SEMANA 8 vs SEMANA 9 ---")

# --- Executando Newton (Semana 8) ---
t0 = time.time()
res_newton, it_newton = rodar_newton_manual(p_init)
t_newton = (time.time() - t0) * 1000

# --- Executando BFGS (Semana 9) ---
# Note: Passamos apenas a função e o gradiente. O BFGS estima a Hessiana!
t0 = time.time()
res_bfgs = minimize(f_scipy, p_init, method='BFGS', jac=g_scipy, tol=1e-6)
t_bfgs = (time.time() - t0) * 1000

# ==========================================
# 4. RELATÓRIO FINAL
# ==========================================
print("\n" + "="*50)
print(f"{'MÉTRICA':<20} | {'NEWTON (S8)':<12} | {'BFGS (S9)':<12}")
print("-" * 50)
print(f"{'Iterações':<20} | {it_newton:<12} | {res_bfgs.nit:<12}")
print(f"{'Tempo Total (ms)':<20} | {t_newton:<12.4f} | {t_bfgs:<12.4f}")
print(f"{'Custo Final':<20} | {f_scipy(res_newton):<12.6f} | {res_bfgs.fun:<12.6f}")
print(f"{'Esforço Matemático':<20} | {'ALTO (Hessiana)'} | {'BAIXO (Grad)'}")
print("="*50)

if res_bfgs.success:
    print(f"\nSucesso! O ponto ótimo encontrado pelo BFGS foi: {res_bfgs.x}")