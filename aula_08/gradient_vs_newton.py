import numpy as np


# 1. Funções Base (Custo, Gradiente e Hessiana)
def custo(x, y):
    return (x - 2) ** 4 + (x - 2 * y) ** 2 + 10


def gradiente(x, y):
    df_dx = 4 * (x - 2) ** 3 + 2 * x - 4 * y
    df_dy = -4 * x + 8 * y
    return np.array([df_dx, df_dy])


def hessiana(x, y):
    d2f_dx2 = 12 * (x - 2) ** 2 + 2
    d2f_dy2 = 8
    d2f_dxdy = -4
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])


# 2. Implementação do Gradiente Descendente (1ª Ordem)
def rodar_gradiente(ponto_inicial, alpha=0.01, tol=1e-6, max_iter=10000):
    ponto = np.array(ponto_inicial, dtype=float)
    it_grad = 0

    for i in range(max_iter):
        it_grad += 1
        g = gradiente(ponto[0], ponto[1])

        if np.linalg.norm(g) < tol:
            break

        ponto = ponto - alpha * g

    return ponto, it_grad


# 3. Implementação do Método de Newton (2ª Ordem)
def rodar_newton(ponto_inicial, tol=1e-6, max_iter=50):
    ponto = np.array(ponto_inicial, dtype=float)
    it_newton = 0

    for i in range(max_iter):
        it_newton += 1
        g = gradiente(ponto[0], ponto[1])
        H = hessiana(ponto[0], ponto[1])

        if np.linalg.norm(g) < tol:
            break

        # Resolução do sistema H * passo = g
        passo = np.linalg.solve(H, g)
        ponto = ponto - passo

    return ponto, it_newton


# --- EXECUÇÃO DO EXPERIMENTO ---
p_start = [0.0, 3.0]

# Executa Gradiente
res_g, cont_g = rodar_gradiente(p_start, alpha=0.02)

# Executa Newton
res_n, cont_n = rodar_newton(p_start)

# 4. Exibição dos Resultados Separados
print("=" * 50)
print("RELATÓRIO DE DESEMPENHO: SEMANA 8")
print("=" * 50)

print(f"MÉTODO DO GRADIENTE DESCENDENTE:")
print(f"  - Resultado: x={res_g[0]:.4f}, y={res_g[1]:.4f}")
print(f"  - Custo Final: {custo(res_g[0], res_g[1]):.8f}")
print(f"  - Total de Iterações: {cont_g}")

print("-" * 30)

print(f"MÉTODO DE NEWTON:")
print(f"  - Resultado: x={res_n[0]:.4f}, y={res_n[1]:.4f}")
print(f"  - Custo Final: {custo(res_n[0], res_n[1]):.8f}")
print(f"  - Total de Iterações: {cont_n}")
print("=" * 50)
