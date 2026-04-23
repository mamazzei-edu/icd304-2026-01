import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# MÉTODO DA RAZÃO ÁUREA (O Maestro)
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


# 1. Definição da Função de Custo e do Gradiente
def custo(x, y):
    return x**2 - 2*x*y + 5*y**2 + 10

def gradiente(x, y):
    df_dx = 2*x - 2*y
    df_dy = -2*x + 10*y
    return np.array([df_dx, df_dy])

# 2. Algoritmo do Gradiente Descendente (Máxima Descida)
def gradiente_descendente(ponto_inicial, alpha, iteracoes=50, tol=1e-5):
    # O ponto atual é um vetor [x, y]
    ponto_atual = np.array(ponto_inicial, dtype=float)
    
    # Lista para guardar o caminho (histórico) para o gráfico
    caminho = [ponto_atual.copy()]
    
    for i in range(iteracoes):
        grad = gradiente(ponto_atual[0], ponto_atual[1])
        
        # Critério de parada: se o gradiente for muito próximo de zero, chegamos ao fundo!
        if np.linalg.norm(grad) < tol:
            print(f"Convergiu na iteração {i}")
            break
            
        # A EQUAÇÃO MÁGICA: Novo Ponto = Ponto Atual - (Passo * Gradiente)
        # ponto_atual = ponto_atual - alpha * grad
        
        #Em vez de usar um ponto alpha fixo, como acima, vamos usar a Razão Áurea para encontrar o melhor alpha a cada iteração!
        # A Razão Áurea é uma técnica de busca unidimensional que nos ajuda a encontrar o melhor passo (alpha) para minimizar a função ao longo da direção do gradiente.
        # Criamos um função temporário para achar o melho passo naquela direção do gradiente
        def g(a):
            p = ponto_atual -a * grad
            return custo(p[0], p[1])
        
        alpha_otimo = golden_section_search(g, 0, 1)  # Buscamos o melhor alpha entre 0 e 1
        ponto_atual = ponto_atual - alpha_otimo * grad
        

        # Salva o novo ponto no histórico
        caminho.append(ponto_atual.copy())
        
    return np.array(caminho)

# 3. Execução do Algoritmo
# Chute inicial ruim: x=4, y=3
ponto_inicial = [4.0, 3.0] 
taxa_aprendizado = 0.0001  # Nosso Passo (Alpha)
# taxa_aprendizado = 0.01  # Passo formiga
# taxa_aprendizado = 0.15  # Passo ideal
# taxa_aprendizado = 0.22  # Zig-zag
# taxa_aprendizado = 0.30  # Divergente
# taxa_aprendizado = 0.50  # Divergente


caminho_percorrido = gradiente_descendente(ponto_inicial, taxa_aprendizado)

ponto_final = caminho_percorrido[-1]
print(f"Ponto Ótimo Encontrado: x = {ponto_final[0]:.4f}, y = {ponto_final[1]:.4f}")
print(f"Custo Mínimo: R$ {custo(ponto_final[0], ponto_final[1]):.2f}")

# 4. Visualização do Caminho Percorrido
# Preparando o terreno (Grid) para o gráfico
x_grid = np.linspace(-1, 5, 100)
y_grid = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = custo(X, Y)

plt.figure(figsize=(10, 8))

# Desenhando as Curvas de Nível (Contour)
cp = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(cp, label='Custo de Resfriamento')

# Extraindo as coordenadas x e y do histórico do algoritmo
x_caminho = caminho_percorrido[:, 0]
y_caminho = caminho_percorrido[:, 1]

# Plotando o caminho do Gradiente Descendente
plt.plot(x_caminho, y_caminho, marker='o', color='red', markersize=4, linestyle='-', linewidth=2, label='Caminho do Algoritmo')

# Marcando Início e Fim
plt.scatter(x_caminho[0], y_caminho[0], color='black', s=100, label='Início', zorder=5)
plt.scatter(x_caminho[-1], y_caminho[-1], color='blue', s=100, marker='*', label='Mínimo Ótimo', zorder=5)

plt.title(f"Gradiente Descendente (Alpha = {taxa_aprendizado})")
plt.xlabel("Temperatura AC (x)")
plt.ylabel("Rotação Exaustor (y)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()