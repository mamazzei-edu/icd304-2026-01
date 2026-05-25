import numpy as np
from sympy import symbols, exp, log, diff, lambdify

# 1. PREPARAÇÃO (O Cérebro Simbólico)
x_s, y_s = symbols('x y')
f_sym = exp(x_s - 2) + exp(y_s - 1) - log(1 + x_s**2 + y_s**2) + (x_s - 2*y_s)**2 + 10
grad_sym = [diff(f_sym, var) for var in (x_s, y_s)]

f_num = lambdify((x_s, y_s), f_sym, 'numpy')
g_num = lambdify((x_s, y_s), grad_sym, 'numpy')

# 2. O ALGORITMO BFGS
def otimizador_bfgs_manual(p0, max_iter=20, tol=1e-5):
    x = np.array(p0, dtype=float)
    n = len(x)
    
    # Inicialização: H começa como a Identidade (Assumimos terreno plano)
    H = np.eye(n)
    
    print(f"{'Iter':<5} | {'Ponto (x, y)':<20} | {'Norma Grad':<10}")
    print("-" * 50)

    for i in range(max_iter):
        g = np.array(g_num(x[0], x[1]))
        norm_g = np.linalg.norm(g)
        
        print(f"{i:<5} | ({x[0]:.4f}, {x[1]:.4f}) | {norm_g:.6f}")
        
        if norm_g < tol:
            print(f"\n✅ Convergiu em {i} iterações!")
            return x

        # A) DIREÇÃO DE BUSCA (Newton aproximado)
        # Em vez de resolver H*p = -g, fazemos apenas uma multiplicação! (O(n^2))
        p = -np.dot(H, g)
        
        # B) BUSCA POR LINHA (Simplificada para o exemplo)
        # Em um solver real, usaríamos Razão Áurea ou Condições de Wolfe aqui
        alpha = 0.1 
        x_novo = x + alpha * p
        g_novo = np.array(g_num(x_novo[0], x_novo[1]))
        
        # C) VETORES DE MEMÓRIA (s e y)
        s = x_novo - x        # Deslocamento na posição
        y = g_novo - g        # Deslocamento no gradiente
        
        # D) ATUALIZAÇÃO DA INVERSA (Fórmula BFGS)
        # rho garante que a matriz continue Positiva Definida
        denominador = np.dot(y, s)
        if denominador > 1e-10:
            rho = 1.0 / denominador
            I = np.eye(n)
            
            # A mágica do O(n^2): Atualização de Posto 2
            A1 = I - rho * np.outer(s, y)
            A2 = I - rho * np.outer(y, s)
            H = np.dot(A1, np.dot(H, A2)) + rho * np.outer(s, s)
        
        # Atualiza o ponto para a próxima rodada
        x = x_novo

    return x

# 3. TESTE DE FOGO
p_inicial = [0.0, 0.0]
resultado = otimizador_bfgs_manual(p_inicial)
print(f"Ponto Ótimo: {resultado}")