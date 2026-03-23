import numpy as np
from scipy.optimize import linprog

# 1. Coeficientes de Custo (Minimização)
c = [550, 700, 720, 850, 980, 1100]

# 2. Matriz de Restrições (A_ub) e Vetor de Limites (b_ub)
# Lembre-se: Inverter o sinal da demanda para atender o padrão <= do solver
A = [
    [-3, -3, -6, -6, -12, -12],  # Demanda Total
    [15000, 2000, 18000, 2500, 22000, 3500],  # Teto de Carbono
    [0, 0, 0, 0, 1, 1],  # Limite de Frota Truck
]

b = [-600, 800000, 20]

# 3. Execução
res = linprog(c, A_ub=A, b_ub=b, method="highs")

# 4. Resultados
if res.success:
    print(f"--- RESULTADO DA OPERAÇÃO BASE ---")
    print(f"Custo Total: R$ {res.fun:.2f}")
    print(f"Alocação (x1 a x6): {np.round(res.x, 2)}")
    print(f"Preços Sombra: {res.ineqlin.marginals}")
else:
    print("Inviável.")
