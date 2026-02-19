import numpy as np
import matplotlib.pyplot as plt

# Configuração do Tamanho do Gráfico
plt.figure(figsize=(10, 8))

# 1. DEFININDO AS RESTRIÇÕES (EQUAÇÕES DE RETA)
# Isolamos x2 para poder plotar: x2 = f(x1)

# Eixo X (x1 - DL) vai de 0 a 80 para visualização
x1 = np.linspace(0, 80, 400)

# Restrição 1: CPU -> 2x1 + 1x2 <= 100  =>  x2 <= 100 - 2x1
y_cpu = 100 - 2 * x1

# Restrição 2: RAM -> 1x1 + 2x2 <= 120  =>  x2 <= (120 - x1) / 2
y_ram = (120 - x1) / 2

# Restrição 3 e 4: Mínimos de Contrato (x1 >= 10, x2 >= 10)
# Serão representados por linhas retas verticais e horizontais

# 2. PLOTANDO AS LINHAS DAS RESTRIÇÕES
plt.plot(x1, y_cpu, label=r'Restrição CPU ($2x_1 + x_2 \leq 100$)', color='blue')
plt.plot(x1, y_ram, label=r'Restrição RAM ($x_1 + 2x_2 \leq 120$)', color='green')
plt.axvline(x=10, color='red', linestyle='--', label=r'Mínimo DL ($x_1 \geq 10$)')
plt.axhline(y=10, color='orange', linestyle='--', label=r'Mínimo DA ($x_2 \geq 10$)')

# 3. IDENTIFICANDO A REGIÃO VIÁVEL
# A região viável é o mínimo das restrições superiores, respeitando os mínimos inferiores
y_min_limit = 10 # Restrição de DA
y_max_feasible = np.minimum(y_cpu, y_ram) # O teto é a menor das duas restrições de hardware

# Preenchendo a região (apenas onde x1 >= 10 e y_max >= 10)
plt.fill_between(x1, y_min_limit, y_max_feasible, 
                 where=(x1 >= 10) & (y_max_feasible >= y_min_limit), 
                 color='gray', alpha=0.3, hatch='//', label='Região Viável')

# 4. PONTO ÓTIMO (Cálculo da Interseção)
# 2x1 + x2 = 100  e  x1 + 2x2 = 120
# Solução: x1 = 80/3 (26.67), x2 = 140/3 (46.67)
opt_x1 = 80/3
opt_x2 = 140/3
lucro_max = 50 * opt_x1 + 30 * opt_x2

plt.plot(opt_x1, opt_x2, 'ro', markersize=10, label='Ponto Ótimo (Vértice)')
plt.annotate(f'Ótimo\n({opt_x1:.1f}, {opt_x2:.1f})\nLucro: R${lucro_max:.0f}', 
             xy=(opt_x1, opt_x2), xytext=(opt_x1+5, opt_x2+10),
             arrowprops=dict(facecolor='black', shrink=0.05))

# 5. LINHA DA FUNÇÃO OBJETIVO (ISOLUCRO) - Opcional para mostrar a inclinação
# Z = 50x1 + 30x2 => x2 = (Z - 50x1) / 30
# Vamos plotar uma linha passando pelo ótimo
y_iso = (lucro_max - 50 * x1) / 30
plt.plot(x1, y_iso, 'k:', linewidth=2, label=f'Reta de Lucro Máximo (Z={lucro_max:.0f})')

# Configurações Finais
plt.xlim(0, 70)
plt.ylim(0, 70)
plt.xlabel('Quantidade de VMs Deep Learning ($x_1$)')
plt.ylabel('Quantidade de VMs Data Analytics ($x_2$)')
plt.title('Solução Gráfica (Método Hillier & Lieberman - Seção 3.1)')
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()