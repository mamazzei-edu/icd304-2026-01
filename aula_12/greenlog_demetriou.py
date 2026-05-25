import numpy as np
from scipy.optimize import differential_evolution

# --- Parâmetros físicos (baseados em Demetriou 2015) ---
P_IT = 10.0  # kW — carga de TI dos servidores
K_f = 3.0  # kW — coeficiente da lei cúbica dos ventiladores
eta_m = 0.65  # eficiência mecânica do ciclo de refrigeração
T_cond = 40.0  # °C — temperatura do condensador
T_max = 35.0  # °C — temperatura máxima admissível na entrada dos racks
rho_cp = 1.2  # kW·s/(m³·°C) — produto ρ·c_p do ar (normalizado)
mu = 5.0  # peso da penalidade térmica


def COP(x):
    """COP do ciclo de refrigeração — Carnot modificado."""
    return eta_m * (x + 273.15) / (T_cond - x + 1e-6)


def demanda_termica(x, y):
    """Diferença entre capacidade de resfriamento e demanda dos racks."""
    capacidade = rho_cp * y * (T_max - x)
    return capacidade - P_IT


def C_greenlog(params):
    """
    Função de custo real — adaptada de Demetriou (2015).

    x : temperatura de fornecimento do ar-condicionado (°C)  [15, 30]
    y : rotação normalizada dos exaustores                   [0.3, 1.0]

    Componentes:
      1. Potência do chiller = (P_IT + fan_power) / COP(x)
      2. Potência dos ventiladores = K_f * y³
      3. Penalidade térmica = μ * deficit²  (zero se resfriamento suficiente)

    Garantia física: C(x,y) > 0 para todo o domínio.
    """
    x, y = params
    fan_power = K_f * y**3
    chiller_power = (P_IT + fan_power) / COP(x)
    deficit = min(0, demanda_termica(x, y))  # só penaliza déficit
    penalty = mu * deficit**2
    return chiller_power + fan_power + penalty


# --- Verificação: mínimo global e comportamento físico ---
bounds = [(15, 28), (0.3, 1.0)]
result = differential_evolution(C_greenlog, bounds, seed=42, maxiter=2000)

print(f"Ótimo global: T_c = {result.x[0]:.1f}°C | Rotação = {result.x[1]:.2f}")
print(f"Custo mínimo = {result.fun:.2f} kW  (sempre > 0 ✓)")

# Verificar que nunca é negativo em uma grade densa
X = np.linspace(15, 28, 200)
Y = np.linspace(0.3, 1.0, 200)
Z = np.array([[C_greenlog([x, y]) for y in Y] for x in X])
print(f"Mínimo absoluto na grade: {Z.min():.3f} kW  (positivo: {Z.min() > 0})")
