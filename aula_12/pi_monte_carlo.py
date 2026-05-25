import random

dentro_circulo = 0
total_pontos = 100000000

for _ in range(total_pontos):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    # Equação do círculo: x^2 + y^2 <= r^2
    if x**2 + y**2 <= 1:
        dentro_circulo += 1

# Estimativa de pi: 4 * (pontos_dentro / total_pontos)
pi_estimado = 4 * dentro_circulo / total_pontos
print(f"Valor estimado de pi: {pi_estimado}")
