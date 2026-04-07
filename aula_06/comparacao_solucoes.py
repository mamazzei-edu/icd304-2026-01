import numpy as np
from scipy.optimize import linprog
import time

# ==========================================
# 1. SETUP DO PROBLEMA (A FUNÇÃO PESADA)
# ==========================================
c = [550, 700, 720, 850, 980, 1100]
A = [
    [-3, -3, -6, -6, -12, -12],
    [0, 0, 0, 0, 1, 1],
    [15000, 2000, 18000, 2500, 22000, 3500],
]
b = [-600, 20, 800000]
bounds = [(0, None)] * 6

# Variável global para contar as chamadas ao Simplex
contador_chamadas = 0

def funcao_pesada(x):
    global contador_chamadas
    contador_chamadas += 1  # Conta cada vez que a função é executada

    custo = []
    for i in range(len(c)):
        if i % 2 == 0:
            custo.append((40 / x + 0.01 * x) * c[i])
        else:
            custo.append((20 / x + 0.025 * x) * c[i])

    res = linprog(custo, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return res.fun


# ==========================================
# 2. AS TRÊS ABORDAGENS DA RAZÃO ÁUREA
# ==========================================


# Abordagem 1: Recursiva sem aproveitamento (com o 'return' arrumado, mas sem reaproveitar pontos)
def golden_ratio_sem_aproveitamento(func, a, b, tol=0.1):
    rho = (np.sqrt(5) - 1) / 2
    d = rho * (b - a)
    x1 = b - d
    x2 = a + d

    # ERRO DE EFICIÊNCIA: Calcula f1 e f2 do zero todas as vezes!
    f1 = func(x1)
    f2 = func(x2)

    if (b - a) < tol:
        return (a + b) / 2

    if f1 < f2:
        return golden_ratio_sem_aproveitamento(func, a, x2, tol)
    else:
        return golden_ratio_sem_aproveitamento(func, x1, b, tol)


# Abordagem 2: Iterativa Padrão (Eficiente - com o 'while')
def golden_ratio_iterativa(func, a, b, tol=0.1):
    rho = (np.sqrt(5) - 1) / 2
    d = rho * (b - a)
    x1 = b - d
    x2 = a + d
    f1 = func(x1)
    f2 = func(x2)

    while (b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1  # REAPROVEITA O PONTO
            x1 = b - rho * (b - a)
            f1 = func(x1)  # 1 ÚNICA CHAMADA
        else:
            a = x1
            x1 = x2
            f1 = f2  # REAPROVEITA O PONTO
            x2 = a + rho * (b - a)
            f2 = func(x2)  # 1 ÚNICA CHAMADA
    return (a + b) / 2


# Abordagem 3: Recursiva Otimizada (Passando o estado adiante)
def golden_ratio_recursiva_otimizada(
    func, a, b, tol=0.1, x1=None, x2=None, f1=None, f2=None
):
    rho = (np.sqrt(5) - 1) / 2

    if x1 is None:
        d = rho * (b - a)
        x1 = b - d
        x2 = a + d
        f1 = func(x1)
        f2 = func(x2)

    if (b - a) < tol:
        return (a + b) / 2

    if f1 < f2:
        novo_b = x2
        novo_d = rho * (novo_b - a)
        novo_x1 = novo_b - novo_d
        novo_f1 = func(novo_x1)  # 1 ÚNICA CHAMADA
        return golden_ratio_recursiva_otimizada(
            func, a, novo_b, tol, novo_x1, x1, novo_f1, f1
        )
    else:
        novo_a = x1
        novo_d = rho * (b - novo_a)
        novo_x2 = novo_a + novo_d
        novo_f2 = func(novo_x2)  # 1 ÚNICA CHAMADA
        return golden_ratio_recursiva_otimizada(
            func, novo_a, b, tol, x2, novo_x2, f2, novo_f2
        )


# ==========================================
# 3. TESTE DE PERFORMANCE - comparando as abordagens
# ==========================================


def rodar_benchmark(nome, funcao_metodo):
    global contador_chamadas
    contador_chamadas = 0  # Zera o contador

    inicio = time.time()
    v_otima = funcao_metodo(funcao_pesada, 40, 90)
    fim = time.time()

    tempo_execucao = fim - inicio

    print(f"--- {nome} ---")
    print(f"Velocidade Ideal: {v_otima:.2f} km/h")
    print(f"Tempo de Execução: {tempo_execucao:.4f} segundos")
    print(f"Chamadas ao Simplex: {contador_chamadas} vezes\n")


print("INICIANDO BENCHMARK DE ALGORITMOS...\n")

rodar_benchmark(
    "Abordagem 1: Recursiva Sem Reaproveitamento", golden_ratio_sem_aproveitamento
)
rodar_benchmark("Abordagem 2: Iterativa Padrão (Com 'while')", golden_ratio_iterativa)
rodar_benchmark(
    "Abordagem 3: Recursiva Otimizada (Passando o Estado)",
    golden_ratio_recursiva_otimizada,
)
