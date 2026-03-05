import numpy as np
import matplotlib.pyplot as plt

# 1. Definindo a Função de Custo (O "Vale")
def f(x, y):
    return x**2 + 2 * y**2

# 2. Criando o "Chão" (Grid de coordenadas)
# Vamos criar uma malha de pontos de -5 a 15 para visualizar bem o vetor
x = np.linspace(-5, 15, 400)
y = np.linspace(-5, 15, 400)
X, Y = np.meshgrid(x, y)

# Calculando a "Altitude" (Erro) para cada ponto do grid
Z = f(X, Y)

# 3. Configurando a Figura
plt.figure(figsize=(10, 8))

# 4. Plotando o Mapa Topográfico (Curvas de Nível)
# Usamos levels para definir quantas "fatias" queremos ver
contorno = plt.contour(X, Y, Z, levels=30, cmap='inferno') ## outros valores magma, plasma, inferno, cvidis, make, rocket, gist_earth, terrain, ocean, gist_stern, gist_rainbow
plt.clabel(contorno, inline=True, fontsize=9) # Coloca os valores de Z nas linhas

# 5. O Ponto Atual (Onde nosso "robô" está)
p_x, p_y = 10, 17
plt.plot(p_x, p_y, 'ro', markersize=8, label=f'Ponto Atual P({p_x}, {p_y})')

# 6. O Vetor Gradiente (Calculado manualmente ou com SymPy)
# Derivadas: df/dx = 2x, df/dy = 4y. No ponto (4,3) -> [?, ?]
grad_x = 20# ?
grad_y = 68# ?

# Plotando a Bússola (A seta do Gradiente)
# Usamos quiver para desenhar o vetor com precisão matemática
plt.quiver(p_x, p_y, grad_x, grad_y, angles='xy', scale_units='xy', scale=1, 
           color='red', width=0.005, label='Gradiente (Sobe a montanha)')

# BÔNUS PEDAGÓGICO: A direção de OTIMIZAÇÃO (Onde o robô deve ir)
# O Gradiente Descendente anda na direção OPOSTA: [-?, -?]
plt.quiver(p_x, p_y, -grad_x, -grad_y, angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.005, label='-Gradiente (Desce a montanha)')

# 7. Estética e Labels
plt.title('A Bússola do Aprendizado: Gradiente vs. Curvas de Nível', fontsize=14)
plt.xlabel('Parâmetro x (Ex: Peso w1)', fontsize=12)
plt.ylabel('Parâmetro y (Ex: Peso w2)', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', alpha=0.5)
plt.legend(loc='upper left')

# Mantém a proporção dos eixos igual para não distorcer o ângulo de 90 graus
plt.axis('equal') 

plt.show()