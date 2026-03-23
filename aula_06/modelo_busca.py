import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Definição da Função de Custo 2D
def f(x, y):
    return x**2 + 2*y**2

# 2. Parâmetros da Busca por Linha
x0, y0 = 1, 1              # Ponto inicial
dx, dy = -2, -4            # Direção de descida (Negativo do Gradiente)

# Função 1D g(alpha) = f(x0 + alpha*dx, y0 + alpha*dy)
def g(alpha):
    return f(x0 + alpha*dx, y0 + alpha*dy)

# 3. Gerando dados para os gráficos
alpha_range = np.linspace(0, 0.6, 100)
alpha_opt = 20/72  # Calculado analiticamente no exercício anterior (~0.277)

# Criando a figura
fig = plt.figure(figsize=(14, 6))

# --- GRÁFICO 1: Superfície 3D e a Direção de Busca ---
ax1 = fig.add_subplot(121, projection='3d')
X = np.linspace(-0.5, 1.5, 40)
Y = np.linspace(-0.5, 1.5, 40)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# Superfície
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
# Linha de busca no chão (z=0) e na superfície
path_x = x0 + alpha_range * dx
path_y = y0 + alpha_range * dy
path_z = f(path_x, path_y)

ax1.plot(path_x, path_y, path_z, color='red', lw=3, label='Caminho da Descida')
ax1.scatter(x0, y0, f(x0, y0), color='black', s=50, label='Ponto Atual P0')
ax1.set_title("A 'Montanha' de Custo e a Direção do Gradiente")
ax1.set_xlabel('Carga CPU (x)')
ax1.set_ylabel('Memória (y)')

# --- GRÁFICO 2: A 'Fatia' 1D (Onde a Razão Áurea atua) ---
ax2 = fig.add_subplot(122)
ax2.plot(alpha_range, g(alpha_range), color='blue', lw=2)
ax2.axvline(alpha_opt, color='red', linestyle='--', label=f'Alpha Ótimo ≈ {alpha_opt:.3f}')
ax2.scatter(alpha_opt, g(alpha_opt), color='red', s=100)

ax2.set_title("A 'Fatia' 1D: Função g(alpha)")
ax2.set_xlabel("Tamanho do Passo (alpha)")
ax2.set_ylabel("Custo Total f(x, y)")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()