import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ==========================================
# 1. DEFINIÇÃO DO PROBLEMA LINEAR (O Operário)
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


def avaliar_frota_simplex(v):
    """
    Dado um valor de velocidade 'v', recalcula os custos não-lineares,
    roda o Simplex e retorna o Custo Mínimo e a Frota Ideal.
    """
    # Funções de multiplicador de custo não-linear
    mult_C = (40 / v) + 0.01 * v
    mult_E = (20 / v) + 0.025 * v

    # Vetor de multiplicadores [C, E, C, E, C, E]
    multiplicadores = np.array([mult_C, mult_E, mult_C, mult_E, mult_C, mult_E])

    # Novo vetor de custos ajustado pela velocidade
    c_dinamico = custo_base * multiplicadores

    # Roda o Simplex
    res = linprog(c_dinamico, A_ub=A, b_ub=b, method="highs")

    if res.success:
        return res.fun, res.x
    else:
        return float("inf"), np.zeros(6)  # Penalidade infinita se for inviável


def custo_objetivo(v):
    """Função 'casca' que a Razão Áurea vai usar (retorna apenas o custo Z)"""
    custo, _ = avaliar_frota_simplex(v)
    return custo


# ==========================================
# 2. MÉTODO DA RAZÃO ÁUREA (O Maestro)
# ==========================================
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


# ==========================================
# 3. EXECUÇÃO E ANÁLISE
# ==========================================
print("Iniciando Otimização Aninhada (Razão Áurea + Simplex)...")
v_otima = golden_section_search(custo_objetivo, 40, 90)
custo_otimo, frota_otima = avaliar_frota_simplex(v_otima)

print(f"\nRESULTADO GLOBAL:")
print(f"Velocidade Ótima Cruzeiro: {v_otima:.2f} km/h")
print(f"Custo Total Mínimo: R$ {custo_otimo:.2f}")
print(f"Frota Ideal nesta velocidade: {np.round(frota_otima, 1)}")

# ==========================================
# 4. GERAÇÃO DE GRÁFICOS (Sensibilidade da Frota)
# ==========================================
v_testes = np.linspace(40, 90, 100)
historico_custos = []
historico_combustao = []
historico_eletrico = []

for v in v_testes:
    c, frota = avaliar_frota_simplex(v)
    historico_custos.append(c)

    # Soma total de viagens de veículos a combustão (índices 0, 2, 4)
    historico_combustao.append(frota[0] + frota[2] + frota[4])
    # Soma total de viagens de veículos elétricos (índices 1, 3, 5)
    historico_eletrico.append(frota[1] + frota[3] + frota[5])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Gráfico 1: A Curva de Custo (Unimodal)
ax1.plot(v_testes, historico_custos, color="purple", lw=2)
ax1.scatter(
    v_otima,
    custo_otimo,
    color="red",
    s=100,
    zorder=5,
    label=f"Mínimo ({v_otima:.1f} km/h)",
)
ax1.set_title("Custo Logístico Total vs Velocidade da Frota")
ax1.set_ylabel("Custo Total (R$)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Gráfico 2: Composição da Frota (As quebras do Simplex)
ax2.plot(
    v_testes, historico_combustao, label="Viagens a Combustão", color="brown", lw=2
)
ax2.plot(v_testes, historico_eletrico, label="Viagens Elétricas", color="green", lw=2)
ax2.set_title("Composição da Frota Ideal ao longo das Velocidades")
ax2.set_xlabel("Velocidade (km/h)")
ax2.set_ylabel("Número de Viagens")
ax2.axvline(v_otima, color="red", linestyle="--", alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
