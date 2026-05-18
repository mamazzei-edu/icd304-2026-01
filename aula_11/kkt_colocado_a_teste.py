import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, LinearConstraint

# =====================================================================
# 1. FUNÇÃO DE CUSTO RUGOSA DA GREENLOG & RESTRIÇÃO
# =====================================================================
def f_cost_global(p):
    x, y = p
    # Base convexa da Semana 10
    base = np.exp(x - 2) + np.exp(y - 1) + np.log(1 + x**2 + y**2) + (x - 2*y)**2
    # Efeito de ressonância e tarifas (Rugosidade não-convexa)
    ressonancia = 2.0 * np.cos(5 * x) * np.sin(5 * y)
    return base + ressonancia

# Restrição de Segurança: x + 2y >= 0
# Para o otimizador Local (SLSQP): g(x) >= 0
con_local = {'type': 'ineq', 'fun': lambda p: p[0] + 2*p[1]}

# Para o otimizador Global: 0 <= 1*x + 2*y <= inf
con_global = LinearConstraint([1, 2], [0], [np.inf])

# Limites do espaço de busca (Bounds) obrigatórios para o algoritmo global
limites = [(-2.0, 4.0), (-2.0, 4.0)]

# =====================================================================
# 2. EXPERIMENTO 1: A FRAGILIDADE DO KKT (Sensibilidade ao Chute Inicial)
# =====================================================================
print("--- TESTE 1: MÉTODOS LOCAIS (KKT / SLSQP) ---")
pontos_iniciais = [
    [0.0, 0.0],
    [1.5, 1.5],
    [-0.5, 0.5],
    [3.0, -1.0]
]

resultados_locais = []

for idx, p_init in enumerate(pontos_iniciais):
    t0 = time.time()
    res = minimize(f_cost_global, p_init, method='SLSQP', constraints=[con_local], tol=1e-6)
    tempo = (time.time() - t0) * 1000
    
    resultados_locais.append({
        'init': p_init,
        'sol': res.x,
        'custo': res.fun,
        'tempo': tempo,
        'sucesso': res.success
    })
    print(f"Chute Inicial {p_init} -> Custo Encontrado: {res.fun:.6f} ({tempo:.2f} ms)")

# =====================================================================
# 3. EXPERIMENTO 2: A ABORDAGEM EVOLUCIONÁRIA GLOBAL (GA)
# =====================================================================
print("\n--- TESTE 2: OTIMIZAÇÃO GLOBAL (Evolução Diferencial) ---")

t0 = time.time()
# O algoritmo espalha uma população aleatória dentro dos 'limites'
res_global = differential_evolution(f_cost_global, bounds=limites, constraints=[con_global], seed=42)
tempo_global = (time.time() - t0) * 1000

print(f"População Inicial: Aleatória dentro do domínio.")
print(f"Melhor Custo Global Encontrado: {res_global.fun:.6f} ({tempo_global:.2f} ms)")
print(f"Melhor Ponto Encontrado: {res_global.x}")

# =====================================================================
# 4. TABELA DE COMPARAÇÃO FINAL
# =====================================================================
print("\n" + "="*80)
print(f"{'ESTRATÉGIA':<25} | {'CHUTE INICIAL':<15} | {'CUSTO FINAL':<12} | {'TEMPO (ms)'}")
print("-" * 80)
for i, r in enumerate(resultados_locais):
    print(f"KKT Local (Tentativa {i+1}) | {str(r['init']):<15} | {r['custo']:<12.6f} | {r['tempo']:.2f}")
print(f"Evolutivo Global (GA)     | {'População':<15} | {res_global.fun:<12.6f} | {tempo_global:.2f}")
print("="*80)