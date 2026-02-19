import numpy as np
from scipy.optimize import linprog

# --- CONFIGURAÇÃO DO PROBLEMA ---

# 1. Função Objetivo
# O linprog faz MINIMIZAÇÃO por padrão.
# Para MAXIMIZAR o lucro (50x1 + 30x2), minimizamos o negativo (-50x1 - 30x2).
c = [-50, -30] 

# 2. Restrições de Desigualdade (Lado Esquerdo - Matriz A)
# Linha 0: CPU (2x1 + 1x2)
# Linha 1: RAM (1x1 + 2x2)
A_ub = [
    [2, 1], 
    [1, 2]
]

# 3. Restrições de Desigualdade (Lado Direito - Vetor b)
# Limites superiores de CPU e RAM
b_ub = [100, 120]

# 4. Limites das Variáveis (Bounds)
# x1 >= 10 e x2 >= 10. O 'None' indica que não há limite superior explícito além das restrições.
x1_bounds = (10, None)
x2_bounds = (10, None)

# --- EXECUÇÃO DO SOLVER ---
# O método 'highs' é o mais moderno incluído no scipy, similar ao Simplex revisado.
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[x1_bounds, x2_bounds], method='highs')

# --- APRESENTAÇÃO DOS RESULTADOS ---
print("=== Resultado da Otimização de Infraestrutura ===")
if res.success:
    print(f"Status: Solução Ótima Encontrada!")
    print(f"Qtd. VMs Deep Learning (x1): {res.x[0]:.2f}")
    print(f"Qtd. VMs Data Analytics (x2): {res.x[1]:.2f}")
    
    # Invertemos o sinal novamente para mostrar o lucro positivo real
    print(f"Lucro Máximo por Hora: R$ {-res.fun:.2f}")
    
    print("\n--- Análise de Recursos (Folgas) ---")
    # Cálculo manual do consumo para discussão
    cpu_usado = 2 * res.x[0] + 1 * res.x[1]
    ram_usada = 1 * res.x[0] + 2 * res.x[1]
    print(f"CPU Utilizada: {cpu_usado:.1f} / 100 (Sobra: {100 - cpu_usado:.1f})")
    print(f"RAM Utilizada: {ram_usada:.1f} / 120 (Sobra: {120 - ram_usada:.1f})")
else:
    print("Não foi possível encontrar uma solução (Verifique as restrições).")