Para implementar a atualização do BFGS em Python, focamos na atualização da **matriz inversa da Hessiana ($H$)**. Em vez de recalcular tudo, nós "corrigimos" a matriz anterior usando os vetores de deslocamento e de variação do gradiente.

Matematicamente, a fórmula que o SciPy e outros solvers usam para atualizar a inversa é a seguinte:

$$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$$

Onde:
* $s_k = x_{k+1} - x_k$ (quanto a posição mudou).
* $y_k = \nabla f_{k+1} - \nabla f_k$ (quanto o gradiente mudou).
* $\rho_k = \frac{1}{y_k^T s_k}$ (um fator de escala escalar).

---

### 🐍 Implementação em Python (NumPy)

Aqui está o trecho de código que realiza essa atualização. Note como usamos o `np.outer` para criar matrizes a partir de vetores, o que mantém a complexidade em $O(n^2)$.

```python
import numpy as np

def atualizar_inversa_bfgs(H_antiga, s, y):
    """
    H_antiga: Matriz de aproximação da inversa da Hessiana (n x n)
    s: Vetor de mudança na posição (x_novo - x_antigo)
    y: Vetor de mudança no gradiente (g_novo - g_antigo)
    """
    n = len(s)
    I = np.eye(n) # Matriz Identidade
    
    # 1. Calcular o escalar rho (1 / produto escalar de y e s)
    # rho garante que a matriz continue positiva definida
    rho = 1.0 / np.dot(y, s)
    
    # 2. Termos intermediários para facilitar a leitura
    # (I - rho * s * y.T)
    A1 = I - rho * np.outer(s, y)
    
    # (I - rho * y * s.T)
    A2 = I - rho * np.outer(y, s)
    
    # 3. A FÓRMULA DE ATUALIZAÇÃO BFGS
    # H_nova = A1 * H_antiga * A2 + rho * (s * s.T)
    H_nova = np.dot(A1, np.dot(H_antiga, A2)) + rho * np.outer(s, s)
    
    return H_nova
```



---

### 🔍 Por que isso é tão eficiente? (Análise do Código)

1.  **Sem `np.linalg.inv`:** Observe que em nenhum momento chamamos a inversão de matriz. Inverter uma matriz é como tentar resolver um quebra-cabeça de $n^3$ peças. Aqui, apenas montamos peças novas sobre a estrutura antiga.
2.  **`np.outer(s, y)`:** Este comando cria uma matriz $n \times n$ a partir de dois vetores. É uma operação simples de multiplicação elemento a elemento ($n^2$ operações).
3.  **Memória de Curvatura:** A matriz `H_nova` agora contém a "memória" de como o terreno curvou durante esse último passo. Na próxima iteração, o passo de busca será $p = -H_{nova} \cdot \nabla f$, que já é a direção de Newton aproximada.

### ⚠️ Dica de Implementação
Para que essa fórmula funcione e a matriz não "exploda" ou se torne instável, é obrigatório que o produto escalar `np.dot(y, s)` seja positivo. Isso é garantido se você fizer uma **Busca por Linha (como a Razão Áurea)** que satisfaça as chamadas *Condições de Wolfe*. Se o passo for muito curto ou aleatório, a aproximação da Hessiana pode "quebrar".

Essa elegância matemática é o que permite que o BFGS seja tão rápido: ele transforma um problema de álgebra linear pesada em uma sequência de manipulações simples de vetores.
