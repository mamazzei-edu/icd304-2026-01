import numpy as np
from scipy.optimize import linprog

# Lembrando: linprog minimiza por padrão. Multiplicamos o lucro por -1.
# Z = 50*DL + 30*DA + 40*ED
c = [-50, -30, -40]

# 2. Matriz de Restrições Técnicas (Lado Esquerdo: A_ub)
# Ordem das colunas: [x1 (DL), x2 (DA), x3 (ED)]
A = [
    [2, 1, 1],  # Restrição 1: Consumo de CPU
    [1, 2, 1],  # Restrição 2: Consumo de RAM
    [0, 1, 2]   # Restrição 3: Consumo de Storage
]

# 3. Vetor de Limites de Capacidade (Lado Direito: b_ub)
b = [150, 160, 100]

# 4. Limites de Não-Negatividade (x >= 0)

# Como não temos mais os contratos mínimos da Semana 1, o limite é de 0 a infinito.
bounds = [(0, None), (0, None), (0, None)]

# 5. Executando o Solver
# O método 'highs' é o padrão atual do SciPy para Programação Linear
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# --- 6. IMPRIMINDO E INTERPRETANDO OS RESULTADOS ---
print("=== Relatório de Otimização do Cluster ===")
if res.success:
    print(f"Status: Otimização Concluída com Sucesso!")
    print(f"Lucro Máximo Estimado: R$ {-res.fun:.2f}\n")
    
    print("--- Plano de Alocação de VMs ---")
    print(f"Deep Learning (x1): {res.x[0]:.2f} instâncias")
    print(f"Data Analytics (x2): {res.x[1]:.2f} instâncias")
    print(f"Engenharia de Dados (x3): {res.x[2]:.2f} instâncias\n")
    
    print("--- Análise de Gargalos (Variáveis de Folga / Slack) ---")
    # A cereja do bolo pedagógico: res.slack mostra o valor final de s1, s2, s3
    folgas = res.slack
    recursos = ["CPU", "RAM", "Storage"]
    
    for i in range(3):
        if np.isclose(folgas[i], 0):
            print(f"{recursos[i]}: GARGALO! Recurso 100% utilizado (Folga = 0)")
        else:
            print(f"{recursos[i]}: OCIOSO. Sobraram {folgas[i]:.2f} unidades.")
else:
    print("Falha na otimização. Verifique o modelo.")