import numpy as np
from scipy.optimize import linprog

# 1. Definindo os Coeficientes da Função Objetivo
# Lembrando: linprog minimiza por padrão. Multiplicamos o lucro por -1.
# Z = 50*DL + 30*DA + 40*ED
c = [-50, -30, -40]

# 2. Matriz de Restrições Técnicas (Lado Esquerdo: A_ub)
# Ordem das colunas: [x1 (DL), x2 (DA), x3 (ED)]
# As restrições técnicas permanecem as mesmas, pois os recursos disponíveis não mudaram.
# As colunas são os coeficientes de consumo de CPU, RAM e Storage para cada tipo de VM.
# São elas que determinam o quanto cada tipo de VM consome de cada recurso, e isso é fundamental para garantir que não ultrapassemos as capacidades do cluster.
A = [
    [2, 1, 1],  # Restrição 1: Consumo de CPU
    [1, 2, 1],  # Restrição 2: Consumo de RAM
    [0, 1, 2],  # Restrição 3: Consumo de Storage
]

# 3. Vetor de Limites de Capacidade (Lado Direito: b_ub)
# As capacidades do cluster permanecem as mesmas, pois não houve alteração nos recursos disponíveis.
b = [160, 160, 100]

# 4. Limites de Não-Negatividade (x >= 0)
# Como não temos mais os contratos mínimos da Semana 1, o limite é de 0 a infinito.
bounds = [(0, None), (0, None), (0, None)]

# 5. Executando o Solver
# O método 'highs' é o padrão atual do SciPy para Programação Linear
# A função linprog é chamada com os coeficientes da função objetivo, as matrizes de restrições e os limites de capacidade.
# O resultado é armazenado na variável 'res', que contém informações sobre a solução encontrada, incluindo o valor ótimo da função objetivo
# e os valores das variáveis de decisão.
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

# --- 6. IMPRIMINDO E INTERPRETANDO OS RESULTADOS ---
# Esta seção é crucial para entender o que a solução encontrada significa na prática.
# O código verifica se a otimização foi bem-sucedida e, em caso afirmativo, imprime o lucro máximo estimado e o plano de alocação de VMs.
# Além disso, a análise de gargalos é feita utilizando as variáveis de folga (slack), que indicam quanto de cada recurso ainda está disponível
# após a alocação das VMs.
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

    ## Agora vamos começar a análise econômica dos resultados, considerando os custos e benefícios associados a cada tipo de VM e o preço-sombra de
    # cada recurso consumido e disponível.
    # O preço-sombra (shadow price) indica o valor marginal de aumentar a capacidade de um recurso.
    # Ele nos ajuda a entender o impacto econômico de cada recurso limitante.
    # Os preços-sombra são fornecidos pela solução do problema de otimização e representam o valor que estamos dispostos a pagar
    # a mais por uma unidade adicional de cada recurso limitante.
    # Os preços-sombra são acessados através da variável 'res.shadow_price'
    print("\n--- Análise Econômica (Preços-Sombra) ---")
    # -- 7 EXTRAINDO O PREÇO SOMBRA (Marginals) --
    # A biblioteca SciPy retorna os valores duais invertidos devido ao padrão de minimização.
    # Usamos o valor absoluto ou multiplicamos por -1 para interpretar como "ganho".
    precos_sombra = -res.ineqlin.marginals  # Preços-sombra para as restrições de capacidade
    recursos = ["CPU", "RAM", "Storage"]
    for i in range(3):
        print(f"{recursos[i]}: Preço-Sombra = R$ {precos_sombra[i]:.2f} por unidade adicional")

    
else:
    print("Falha na otimização. Verifique o modelo.")
