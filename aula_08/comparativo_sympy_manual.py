import numpy as np
import time
from sympy import symbols, hessian, lambdify

# --- SETUP SIMBÓLICO ---
x, y = symbols('x y')
f_sym = (x - 2)**4 + (x - 2*y)**2
H_sym = hessian(f_sym, (x, y))

# Gerando a função numérica "TURBO" com lambdify
hessiana_fast = lambdify((x, y), H_sym, 'numpy')

# --- SETUP MANUAL (Para comparação) ---
def hessiana_manual(x_val, y_val):
    return np.array([[12*(x_val-2)**2 + 2, -4], [-4, 8]])

# --- BENCHMARK DE AVALIAÇÃO ---
n_testes = 10000
p = (0.5, 1.5)

# 1. Teste SymPy Puro (Muito Lento)
t0 = time.time()
for _ in range(100): # Apenas 100 pois é muito lento
    res = H_sym.subs({x: p[0], y: p[1]})
t_subs = (time.time() - t0) / 100

# 2. Teste Manual (Escrito à mão)
t0 = time.time()
for _ in range(n_testes):
    res = hessiana_manual(p[0], p[1])
t_manual = (time.time() - t0) / n_testes

# 3. Teste Lambdify (Otimizado)
t0 = time.time()
for _ in range(n_testes):
    res = hessiana_fast(p[0], p[1])
t_fast = (time.time() - t0) / n_testes

print(f"{'Método':<25} | {'Tempo Médio (s)':<20}")
print("-" * 50)
print(f"{'SymPy (subs)':<25} | {t_subs:.8f}")
print(f"{'Manual (Numpy)':<25} | {t_manual:.8f}")
print(f"{'SymPy (lambdify)':<25} | {t_fast:.8f}")