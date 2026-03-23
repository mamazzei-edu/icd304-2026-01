import numpy as np
from scipy.optimize import linprog

# 1. Definindo os Coeficientes da Função Objetivo
# Custos por 100km de cada tipo de caminhão
c = [550, 720, 980]

# 2. Matriz de Restrições Técnicas (Lado Esquerdo: A_ub)
# Ordem das colunas: [x1 (VUC), x2 (TOCO), x3 (Truck)]
A = [
    [-3, -6, -12],  # Qtde transportada por tipo de caminhão
    [0, 0, 1],  # Limite de 20 trucks
]

# 3. Vetor de Limites de Capacidade (Lado Direito: b_ub)
b = [-600, 20]

# 4. Executando o Solver
# O método 'highs' é o padrão atual do SciPy para Programação Linear
# A função linprog é chamada com os coeficientes da função objetivo, as matrizes de restrições e os limites de capacidade.
# O resultado é armazenado na variável 'res', que contém informações sobre a solução encontrada, incluindo o valor ótimo da função objetivo
# e os valores das variáveis de decisão.
res = linprog(c, A_ub=A, b_ub=b, method="highs")

# IMPRIMINDO E INTERPRETANDO OS RESULTADOS ---
# Esta seção é crucial para entender o que a solução encontrada significa na prática.
# O código verifica se a otimização foi bem-sucedida e, em caso afirmativo, imprime o lucro máximo estimado e o plano de alocação de VMs.
# Além disso, a análise de gargalos é feita utilizando as variáveis de folga (slack), que indicam quanto de cada recurso ainda está disponível
# após a alocação das VMs.
print("=== Relatório de Otimização do Transporte ===")
if res.success:
    print(f"Status: Otimização Concluída com Sucesso!")
    print(f"Total de Custos: R$ {res.fun:.2f}\n")

    print("--- Plano de Alocação de VMs ---")
    print(f"Vuc (x1): {res.x[0]:.2f} instâncias")
    print(f"Toco (x2): {res.x[1]:.2f} instâncias")
    print(f"Truck (x3): {res.x[2]:.2f} instâncias\n")

    print("--- Análise de Gargalos (Variáveis de Folga / Slack) ---")
    precos_sombra = res.ineqlin.marginals  # Aqui estão os Preços Sombra
    folgas = res.slack                     # Aqui vemos se há sobra (folga)
    
    print(f"Solução Ótima (Viagens): {np.round(res.x, 2)}")
    print(f"Custo Total: R$ {res.fun:.2f}")
    print("-" * 30)
    
    print(f"Preço Sombra - Demanda (600t): {precos_sombra[0]:.2f}")
    print(f"Preço Sombra - Limite Truck (20un): {precos_sombra[1]:.2f}")
    print("-" * 30)
    
    print(f"Folga na Demanda: {folgas[0]} t")
    print(f"Folga na Frota Truck: {folgas[1]} un")
else:
    print("Falha na otimização. Verifique o modelo.")
