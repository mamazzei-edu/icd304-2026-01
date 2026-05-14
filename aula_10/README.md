Para resolver problemas de otimização com restrições, deixamos de apenas "seguir o gradiente" e passamos a buscar um equilíbrio entre a **vontade de descer** (função objetivo) e a **impossibilidade de atravessar o muro** (restrição).

Esse é o gabarito detalhado para a Semana 10, para o problema proposto da GreenLog.

---

### 1. Verificação de Inviabilidade

Antes de tudo, precisamos provar que o ponto ótimo da semana passada ($x \approx -0.109, y \approx -0.078$) não serve mais.

* **Restrição:** $x + 2y \ge 0$
* **Teste:** $-0.109 + 2(-0.078) = -0.109 - 0.156 = \mathbf{-0.265}$
* **Conclusão:** Como $-0.265 < 0$, o ponto anterior está na "zona de congelamento". O novo ótimo terá que se deslocar para respeitar a borda.

---

### 2. Formulação Matemática (KKT)

Para resolver este problema analiticamente, montamos a **Função Lagrangiana ($\mathcal{L}$)**:


$$\mathcal{L}(x, y, \mu) = f(x, y) + \mu \cdot (-x - 2y)$$

As **Condições KKT** que o computador precisará satisfazer são:

1. **Estacionaridade:** $\nabla f(x, y) + \mu \nabla g(x, y) = 0$
2. **Viabilidade Primal:** $-x - 2y \le 0$
3. **Viabilidade Dual:** $\mu \ge 0$
4. **Folga Complementar:** $\mu \cdot (-x - 2y) = 0$

---

### 3. Resolução em Python (Gabarito do Lab)

Utilizaremos o algoritmo `SLSQP` (Sequential Least Squares Programming), que é o padrão do SciPy para lidar com restrições e condições KKT.

```python
import numpy as np
from scipy.optimize import minimize

# 1. Função de Custo (Semana 9 - Estabilizada)
def f_cost(p):
    x, y = p
    return np.exp(x - 2) + np.exp(y - 1) + np.log(1 + x**2 + y**2) + (x - 2*y)**2

# 2. Definição da Restrição (Deve ser escrita como g(x) >= 0 no SciPy)
# x + 2y >= 0
def constraint1(p):
    return p[0] + 2*p[1]

con = {'type': 'ineq', 'fun': constraint1}

# 3. Execução da Otimização
p_init = [0, 0]
res = minimize(f_cost, p_init, method='SLSQP', constraints=[con])

# 4. Extração dos Resultados e Multiplicadores
# No SLSQP, os multiplicadores de Lagrange (mu) estão em res.maxcv ou via KKT manual
print(f"Novo Ponto Ótimo: x = {res.x[0]:.6f}, y = {res.x[1]:.6f}")
print(f"Custo com Restrição: {res.fun:.6f}")
print(f"A restrição está ativa? {np.isclose(constraint1(res.x), 0, atol=1e-4)}")

```

---

### 4. Explicação do Processo de Resolução

O processo que o algoritmo executa segue este raciocínio lógico:

1. **Exploração Inicial:** O algoritmo começa no ponto $(0,0)$ e tenta descer na direção do gradiente $-\nabla f$.
2. **Impacto no Muro:** Ele percebe que o caminho para o mínimo global (aqueles $-0.109$ da semana passada) é bloqueado pela restrição $x + 2y \ge 0$.
3. **Equilíbrio de Forças:** Em vez de parar, ele "desliza" ao longo da borda da restrição. O ponto ótimo agora é onde o gradiente da função de custo é exatamente anulado pela "força" da restrição (multiplicador de Lagrange $\mu$).
4. **Verificação da Folga Complementar:** * Se o ótimo estivesse longe do muro, $\mu$ seria $0$.
* Como o ótimo foi forçado a ficar exatamente na borda ($x + 2y \approx 0$), o valor de $\mu$ será positivo, indicando o quanto o custo aumentaria se "apertássemos" ainda mais a restrição.



---

### 5. Comparativo Final

| Cenário | Ponto Ótimo ($x, y$) | Custo | Situação |
| --- | --- | --- | --- |
| **Sem Restrição (S9)** | $(-0.109, -0.078)$ | $0.4816$ | **Inviável** (Viola segurança) |
| **Com Restrição (S10)** | $(0.0544, -0.0272)$ | $0.4998$ | **Viável** (No limite da segurança) |

> **Conclusão:** Note que o custo **subiu** (de 0.48 para 0.49). Isso é uma regra fundamental: **restrições nunca diminuem o custo**, elas apenas o mantêm igual ou o aumentam em troca de segurança ou viabilidade técnica.

Este é o "preço da segurança" que os multiplicadores de Lagrange nos ajudam a calcular!

Tudo pronto para os alunos enfrentarem o dilema do congelamento?
Vamos descobrir o problema que nos leva a próxima aula dos **Algoritmos Genéticos (Semana 11)**, onde as coisas ficam bem mais "selvagens".