Para que o **BFGS** funcione com estabilidade industrial, a escolha do tamanho do passo ($\alpha$) não pode ser aleatória. Se o passo for muito grande ou muito curto, a aproximação da Hessiana pode "quebrar" (deixar de ser positiva definida).

Vamos conhecer as duas abordagens:
- a **Razão Áurea** (que busca o mínimo exato na linha) e 
- as **Condições de Wolfe** (que buscam um passo "bom o suficiente" para garantir a convergência).

---

### 1. BFGS com Razão Áurea (Busca por Linha Exata)
A Razão Áurea é uma busca "cega": ela não olha para o gradiente, apenas para os valores da função. Ela tenta encontrar o $\alpha$ que minimiza $f(x_k + \alpha p_k)$ o máximo possível antes de atualizar a Hessiana.

```python
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
```

---

### 2. Condições de Wolfe (Busca por Linha Inexata)
Diferente da Razão Áurea, as Condições de Wolfe não tentam achar o "fundo do poço". Elas aceitam qualquer passo que satisfaça dois critérios de segurança. Isso é muito mais eficiente computacionalmente.

#### As Duas Regras de Ouro:
1.  **Condição de Armijo (Decréscimo Suficiente):** O passo não pode ser tão grande que o custo não diminua proporcionalmente à inclinação.
    $$f(x_k + \alpha p_k) \le f(x_k) + c_1 \alpha \nabla f_k^T p_k$$
2.  **Condição de Curvatura:** Garante que o gradiente no novo ponto não seja tão íngreme quanto o anterior. **Isso é o que garante que $y_k^T s_k > 0$**, mantendo a matriz Hessiana estável.
    $$\nabla f(x_k + \alpha p_k)^T p_k \ge c_2 \nabla f_k^T p_k$$



---

### 🚀 O Algoritmo Integrado: Newton-BFGS com Wolfe vs. GSS

```python
from scipy.optimize import line_search # SciPy já implementa Wolfe de forma robusta

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
# f_green = lambda p: (p[0]-2)**4 + (p[0]-2*p[1])**2
# g_green = lambda p: np.array([4*(p[0]-2)**3 + 2*(p[0]-2*p[1]), -4*(p[0]-2*p[1])])
# bfgs_wolfe_vs_gss(f_green, g_green, [0, 3], metodo='wolfe')
```

---

### 📊 Comparativo Crítico

* **Com Razão Áurea:** O algoritmo é mais "teimoso". Ele gasta muito tempo processando a função para achar o $\alpha$ perfeito em cada iteração. É bom para funções muito simples, mas desperdiça processamento em funções complexas.
* **Com Condições de Wolfe:** É o comportamento "inteligente". Ele dá um passo que "parece bom o suficiente" e deixa a atualização da Hessiana compensar as imprecisões no próximo turno. 

**Por que Wolfe é vital para o BFGS?**
Sem a segunda condição de Wolfe (curvatura), o valor de `ys = np.dot(y, s)` poderia ser negativo ou zero. Se isso acontecesse, a matriz $H$ perderia sua propriedade de "mínimo" e o algoritmo poderia começar a subir a montanha em vez de descer.
