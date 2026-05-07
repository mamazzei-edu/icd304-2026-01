from scipy.optimize import line_search # SciPy já implementa Wolfe de forma robusta
import numpy as np

def gss_step(f, p_atual, direcao, a=0, b=1, tol=1e-5):
    """Busca o alpha ideal usando a Razão Áurea"""
    rho = (np.sqrt(5) - 1) / 2
    
    # Função univariável dependente apenas de alpha
    phi = lambda alpha: f(p_atual + alpha * direcao)
    
    d = rho * (b - a)
    x1, x2 = b - d, a + d
    f1, f2 = phi(x1), phi(x2)
    
    while (b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - rho * (b - a)
            f1 = phi(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + rho * (b - a)
            f2 = phi(x2)
    return (a + b) / 2

def bfgs_wolfe_vs_gss(f, grad, p0, metodo='wolfe'):
    x = np.array(p0, dtype=float)
    n = len(x)
    H = np.eye(n) # Identidade Inicial
    
    for i in range(20):
        g = grad(x)
        if np.linalg.norm(g) < 1e-5: break
        
        direcao = -np.dot(H, g) # Passo de busca
        
        # --- ESCOLHA DO MÉTODO DE PASSO ---
        if metodo == 'wolfe':
            # O line_search do scipy verifica as condições de Armijo e Curvatura
            res = line_search(f, grad, x, direcao)
            alpha = res[0] if res[0] is not None else 0.1
        else:
            alpha = gss_step(f, x, direcao)
        
        # Atualização
        x_novo = x + alpha * direcao
        s = x_novo - x
        y = grad(x_novo) - g
        
        # Atualização BFGS da Inversa (H)
        ys = np.dot(y, s)
        if ys > 1e-10: # Só atualiza se a condição de curvatura permitir
            rho = 1.0 / ys
            I = np.eye(n)
            A1 = I - rho * np.outer(s, y)
            A2 = I - rho * np.outer(y, s)
            H = np.dot(A1, np.dot(H, A2)) + rho * np.outer(s, s)
        
        x = x_novo
        print(f"Iter {i} | Metodo: {metodo} | Alpha: {alpha:.4f} | Custo: {f(x):.6f}")
    
    return x

# Exemplo de uso (Função GreenLog simplificada para teste)
f_green = lambda p: (p[0]-2)**4 + (p[0]-2*p[1])**2
g_green = lambda p: np.array([4*(p[0]-2)**3 + 2*(p[0]-2*p[1]), -4*(p[0]-2*p[1])])
bfgs_wolfe_vs_gss(f_green, g_green, [0, 3], metodo='wolfe')
bfgs_wolfe_vs_gss(f_green, g_green, [0, 3], metodo='gss')