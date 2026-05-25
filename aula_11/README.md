## O custo difícil para KKT ##

Imagine que na GreenLog, o custo real é dado por: 

$$f(x, y)_{\text{global}} = f(x, y)_{\text{S10}} + \cos(5x) \cdot \sin(5y)$$


 Isso transforma o "vale liso" em um terreno altamente rugoso (não-convexo), repleto de falsos mínimos locais (armadilhas).
 
 Mantemos a restrição de segurança da semana passada ($x + 2y \ge 0$) para vermos como o KKT lida com ela localmente, enquanto o algoritmo evolucionário a resolve globalmente.


```python
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, LinearConstraint

# ==========================================
# 1. A NOVA FUNÇÃO DE CUSTO RUGOSA (GreenLog)
# ==========================================
def f_rugged_cost(p):
    x, y = p
    # Base da Semana 10
    base = np.exp(x - 2) + np.exp(y - 1) + np.log(1 + x**2 + y**2) + (x - 2*y)**2
    # Ruído de Alta Frequência (Cria os mínimos locais/armadilhas)
    rugosidade = 2.0 * np.cos(5 * x) * np.sin(5 * y)
    return base + rugosidade

# ==========================================
# 2. DEFINIÇÃO DA RESTRIÇÃO (x + 2y >= 0)
# ==========================================
# Para o otimizador Local (SLSQP)
con_local = {'type': 'ineq', 'fun': lambda p: p[0] + 2*p[1]}

# Para o otimizador Global (Differential Evolution)
# Restrição Linear: 0 <= 1*x + 2*y <= inf
con_global = LinearConstraint([1, 2], [0], [np.inf])

# Otimizadores globais precisam de um "espaço de busca" (Bounds) para espalhar a população
limites = [(-2.0, 4.0), (-2.0, 4.0)]

# ==========================================
# 3. EXECUÇÃO 1: OTIMIZAÇÃO LOCAL (KKT / SLSQP)
# ==========================================
# Simulando um aluno que escolheu um ponto inicial "azarado"
p_inicial = [1.5, 1.5]

t0 = time.time()
res_local = minimize(f_rugged_cost, p_inicial, method='SLSQP', constraints=[con_local], tol=1e-6)
tempo_local = (time.time() - t0) * 1000

# ==========================================
# 4. EXECUÇÃO 2: OTIMIZAÇÃO GLOBAL (Algoritmo Evolucionário)
# ==========================================
# O Differential Evolution é o primo contínuo do Algoritmo Genético clássico
t0 = time.time()
res_global = differential_evolution(f_rugged_cost, bounds=limites, constraints=[con_global], seed=42)
tempo_global = (time.time() - t0) * 1000

# ==========================================
# 5. RELATÓRIO DO CONFRONTOS DE PARADIGMAS
# ==========================================
print("\n" + "="*70)
print(f"{'MÉTRICA':<25} | {'KKT / SLSQP (LOCAL)':<20} | {'GA / EVOLUTIVO (GLOBAL)':<20}")
print("-" * 70)
print(f"{'Ponto Inicial':<25} | {str(p_inicial):<20} | {'Aleatório (População)':<20}")
print(f"{'Custo Encontrado':<25} | {res_local.fun:<20.6f} | {res_global.fun:<20.6f}")
print(f"{'Ponto Ótimo (x*)':<25} | {res_local.x[0]:<20.6f} | {res_global.x[0]:<20.6f}")
print(f"{'Ponto Ótimo (y*)':<25} | {res_local.x[1]:<20.6f} | {res_global.x[1]:<20.6f}")
print(f"{'Tempo de Execução':<25} | {tempo_local:<17.2f} ms | {tempo_global:<17.2f} ms")
print(f"{'Status da Solução':<25} | {'PRESO EM MÍN. LOCAL'} | {'MÍNIMO GLOBAL REAL'}")
print("="*70)

```
