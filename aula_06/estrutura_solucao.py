import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ==========================================
# Passo 1. DEFINIÇÃO DO PROBLEMA LINEAR (O Operário)
# ==========================================
# Matriz de Restrições A_ub e vetor b_ub (Fixos)
A = [
    [-3, -3, -6, -6, -12, -12],  # Demanda (>= 600t invertido)
    [15000, 2000, 18000, 2500, 22000, 3500],  # Teto de Carbono (<= 800.000g)
    [0, 0, 0, 0, 1, 1],  # Máximo de 20 Trucks
]
b = [-600, 800000, 20]

# Custos base da tabela [VUC-C, VUC-E, Toco-C, Toco-E, Truck-C, Truck-E]
custo_base = np.array([550, 700, 720, 850, 980, 1100])

# PASSO 2: A Função Objetivo que a Razão Áurea vai testar
def custo_total_simulado(v):
    # 1. Calcula os multiplicadores para a velocidade 'v'
    mult_C = (40 / v) + 0.01 * v
    mult_E = (20 / v) + 0.025 * v
    p_c = 0.5 * 1.225 * 0,38 * 2 * v**3
    p_e = 0.5 * 1.225 * 0,21 * 2 * v**3

    # Aplica mult_C nas posições pares e mult_E nas ímpares
    c_dinamico = custo_base * [mult_C, mult_E, mult_C, mult_E, mult_C, mult_E]
    c_dinamico = custo_base * [mult_C, mult_E, mult_C, mult_E, mult_C, mult_E]

    # 3. Roda o Simplex (A_ub e b_ub continuam os mesmos!)
    res = linprog(c_dinamico, A_ub=A, b_ub=b, method="highs")

    # 4. Retorna o custo total Z (que a Razão Áurea quer minimizar)
    return res.fun, res.x


# PASSO 3: Aqui a Razão Áurea "envelopa" o Simplex
# Vocês deverão utilizar a função golden_section_search que aprenderam,
# mas passando a função 'custo_total_simulado' como alvo.
def golden_section_search(f, a, b, tol=0.01):
    rho = (np.sqrt(5) - 1) / 2
    d = rho * (b - a)
    x1 = b - d
    x2 = a + d

    f1 = f(x1)
    f2 = f(x2)

    while (b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - rho * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + rho * (b - a)
            f2 = f(x2)

    return (a + b) / 2

print("Velocidade Ótima:", golden_section_search(custo_total_simulado, 40, 90))
print("Frota ideal para essa velocidade:", custo_total_simulado(golden_section_search(custo_total_simulado, 40, 90))[1])
