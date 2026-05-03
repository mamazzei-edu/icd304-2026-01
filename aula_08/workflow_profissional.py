import numpy as np
import time
from sympy import symbols, hessian, lambdify, diff

# ==========================================
# 1. SETUP SIMBÓLICO (O "CÉREBRO")
# ==========================================
x_sym, y_sym = symbols('x y')

# Definimos a função de custo uma única vez
f_sym = (x_sym - 2)**4 + (x_sym - 2*y_sym)**2

# O SymPy calcula as derivadas automaticamente
grad_sym = [diff(f_sym, var) for var in (x_sym, y_sym)]
hess_sym = hessian(f_sym, (x_sym, y_sym))

# O lambdify "compila" as expressões para funções NumPy ultra-rápidas
custo_func = lambdify((x_sym, y_sym), f_sym, 'numpy')
grad_func  = lambdify((x_sym, y_sym), grad_sym, 'numpy')
hess_func  = lambdify((x_sym, y_sym), hess_sym, 'numpy')

# ==========================================
# 2. ALGORITMO DE NEWTON (O "MOTOR")
# ==========================================
def solver_newton_auto(ponto_inicial, tol=1e-7, max_iter=20):
    ponto = np.array(ponto_inicial, dtype=float)
    historico = []
    
    print(f"{'Iter':<5} | {'Ponto (x, y)':<20} | {'Custo':<10}")
    print("-" * 50)
    
    for i in range(max_iter):
        # Avaliação numérica rápida usando as funções lambdificadas
        c_atual = custo_func(ponto[0], ponto[1])
        g_num = np.array(grad_func(ponto[0], ponto[1]))
        H_num = np.array(hess_func(ponto[0], ponto[1]))
        
        print(f"{i:<5} | ({ponto[0]:.4f}, {ponto[1]:.4f}) | {c_atual:.8f}")
        historico.append(ponto.copy())
        
        # Critério de parada (norma do gradiente próxima de zero)
        if np.linalg.norm(g_num) < tol:
            print("-" * 50)
            print(f"✅ Convergiu em {i} iterações!")
            return ponto, i
            
        # Passo de Newton: Resolve H * s = g => s = H^-1 * g
        try:
            passo = np.linalg.solve(H_num, g_num)
            ponto = ponto - passo
        except np.linalg.LinAlgError:
            print("❌ Erro: Hessiana singular (não invertível).")
            return None, i
            
    return ponto, max_iter

# ==========================================
# 3. TESTE DE PERFORMANCE
# ==========================================
p_inicio = [0.0, 3.0]
t0 = time.time()
resultado, iters = solver_newton_auto(p_inicio)
t_final = (time.time() - t0) * 1000

print(f"Resultado Final: x = {resultado[0]:.4f}, y = {resultado[1]:.4f}")
print(f"Tempo de execução: {t_final:.2f} ms")