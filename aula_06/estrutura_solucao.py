# PASSO 1: A Função Objetivo que a Razão Áurea vai testar
def custo_total_simulado(v):
    # 1. Calcula os multiplicadores para a velocidade 'v'
    mult_C = (40 / v) + 0.01 * v
    mult_E = (20 / v) + 0.025 * v

    # 2. Atualiza o vetor de custos base 'c'
    # Base: [VUC-C, VUC-E, Toco-C, Toco-E, Truck-C, Truck-E]
    custo_base = np.array([550, 700, 720, 850, 980, 1100])

    # Aplica mult_C nas posições pares e mult_E nas ímpares
    c_dinamico = custo_base * [mult_C, mult_E, mult_C, mult_E, mult_C, mult_E]

    # 3. Roda o Simplex (A_ub e b_ub continuam os mesmos!)
    res = linprog(c_dinamico, A_ub=A, b_ub=b, method="highs")

    # 4. Retorna o custo total Z (que a Razão Áurea quer minimizar)
    return res.fun


# PASSO 2: Aqui a Razão Áurea "envelopa" o Simplex
# Vocês deverão utilizar a função golden_section_search que aprenderam,
# mas passando a função 'custo_total_simulado' como alvo.
