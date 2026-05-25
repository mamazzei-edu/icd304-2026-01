import numpy as np
from scipy.optimize import minimize


# 1. Função de Custo (Semana 9 - Estabilizada)
def f_cost(p):
    x, y = p
    return (
        np.exp(x - 2) + np.exp(y - 1) + np.log(1 + x**2 + y**2) + (x - 2 * y) ** 2 + 10
    )


# 2. Definição da Restrição (Deve ser escrita como g(x) >= 0 no SciPy)
# x + 2y >= 0
def constraint1(p):
    return p[0] + 2 * p[1]


con = {"type": "ineq", "fun": constraint1}

# 3. Execução da Otimização
p_init = [3, 2]
res = minimize(f_cost, p_init, method="SLSQP", constraints=[con])

# 4. Extração dos Resultados e Multiplicadores
# No SLSQP, os multiplicadores de Lagrange (mu) estão em res.maxcv ou via KKT manual
print(f"Novo Ponto Ótimo: x = {res.x[0]:.6f}, y = {res.x[1]:.6f}")
print(f"Custo com Restrição: {res.fun:.6f}")
print(f"A restrição está ativa? {np.isclose(constraint1(res.x), 0, atol=1e-4)}")
