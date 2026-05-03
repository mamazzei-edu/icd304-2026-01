Esta é a versão definitiva e mais sofisticada do algoritmo para o laboratório da **Semana 8**. Nela, integramos o poder do **SymPy** para a derivação automática com a velocidade do **NumPy** via `lambdify`.

Esta abordagem elimina a necessidade de calcular o Gradiente e a Hessiana manualmente, permitindo que foquemos na lógica da convergência.

---

### 🚀 Newton's Method: Versão Integrada (SymPy + NumPy)

```python
import numpy as np
import time
from sympy import symbols, hessian, lambdify, diff

# ==========================================
# 1. SETUP SIMBÓLICO (O "CÉREBRO")
# ==========================================
x_sym, y_sym = symbols('x y')

# Definimos a função de custo uma única vez
f_sym = (x_sym - 2)**4 + (x_sym - 2*y_sym)**2

# O SymPy calcula as derivadas automaticamente
grad_sym = [diff(f_sym, var) for var in (x_sym, y_sym)]
hess_sym = hessian(f_sym, (x_sym, y_sym))

# O lambdify "compila" as expressões para funções NumPy ultra-rápidas
custo_func = lambdify((x_sym, y_sym), f_sym, 'numpy')
grad_func  = lambdify((x_sym, y_sym), grad_sym, 'numpy')
hess_func  = lambdify((x_sym, y_sym), hess_sym, 'numpy')

# ==========================================
# 2. ALGORITMO DE NEWTON (O "MOTOR")
# ==========================================
def solver_newton_auto(ponto_inicial, tol=1e-7, max_iter=20):
    ponto = np.array(ponto_inicial, dtype=float)
    historico = []
    
    print(f"{'Iter':<5} | {'Ponto (x, y)':<20} | {'Custo':<10}")
    print("-" * 50)
    
    for i in range(max_iter):
        # Avaliação numérica rápida usando as funções lambdificadas
        c_atual = custo_func(ponto[0], ponto[1])
        g_num = np.array(grad_func(ponto[0], ponto[1]))
        H_num = np.array(hess_func(ponto[0], ponto[1]))
        
        print(f"{i:<5} | ({ponto[0]:.4f}, {ponto[1]:.4f}) | {c_atual:.8f}")
        historico.append(ponto.copy())
        
        # Critério de parada (norma do gradiente próxima de zero)
        if np.linalg.norm(g_num) < tol:
            print("-" * 50)
            print(f"✅ Convergiu em {i} iterações!")
            return ponto, i
            
        # Passo de Newton: Resolve H * s = g => s = H^-1 * g
        try:
            passo = np.linalg.solve(H_num, g_num)
            ponto = ponto - passo
        except np.linalg.LinAlgError:
            print("❌ Erro: Hessiana singular (não invertível).")
            return None, i
            
    return ponto, max_iter

# ==========================================
# 3. TESTE DE PERFORMANCE
# ==========================================
p_inicio = [0.0, 3.0]
t0 = time.time()
resultado, iters = solver_newton_auto(p_inicio)
t_final = (time.time() - t0) * 1000

print(f"Resultado Final: x = {resultado[0]:.4f}, y = {resultado[1]:.4f}")
print(f"Tempo de execução: {t_final:.2f} ms")
```

---

### 📝 Por que esta versão é superior para os alunos?

1.  **Confiabilidade Matemática:** Nós não corremos o risco de errar um sinal ou uma regra da cadeia ao derivar a Hessiana. O SymPy garante a precisão algébrica.
2.  **Separação de Camadas:**
    * **Camada Simbólica:** Onde definimos a física/matemática do problema.
    * **Camada Numérica:** Onde o algoritmo de Newton "corre" com performance máxima.
3.  **Flexibilidade Extrema:** Se quisermos mudar a função de custo para um polinómio de 6ª ordem ou adicionar funções trigonométricas no próximo laboratório, basta alterar a linha `f_sym` e **todo o resto do código continuará a funcionar perfeitamente** sem que tenhamos de recalcular uma única derivada.
4.  **Preparação Profissional:** Este é o *workflow* padrão em bibliotecas de otimização modernas (como PyTorch ou JAX), onde a diferenciação automática é a norma.



Aqui está a **Folha de Dicas (Cheat Sheet)** desenhada para os seus alunos. O objetivo é que ela seja um guia de "primeiros socorros" quando o código não rodar ou quando a matemática simbólica do SymPy parecer confusa.

---

# 💡 Folha de Dicas: SymPy & Otimização (Semana 8)

Este guia ajudará você a navegar pela integração entre **Matemática Simbólica (SymPy)** e **Cálculo Numérico (NumPy)** no Método de Newton.

---

## 1. O básico do SymPy: Não esqueça os Símbolos!
O SymPy não entende variáveis comuns do Python como variáveis matemáticas, a menos que você as declare explicitamente.

* **O Erro:** `NameError: name 'x' is not defined`
* **A Solução:** Sempre inicie seu script declarando seus símbolos:
    ```python
    from sympy import symbols
    x, y = symbols('x y') # Agora o Python sabe que x e y são variáveis algébricas
    ```

---

## 2. Derivadas Automáticas (O fim do erro de cálculo)
Em vez de usar a regra da cadeia na mão, use o SymPy.

* **Gradiente:** Use `diff(f, variavel)`.
* **Hessiana:** Use `hessian(f, (lista_de_variaveis))`.
    * *Dica de Ouro:* A ordem na lista `(x, y)` define a ordem das linhas e colunas na matriz final.



---

## 3. O "Pulo do Gato": `lambdify`
O `lambdify` transforma uma expressão do SymPy (lenta/simbólica) em uma função do NumPy (rápida/numérica).

* **A Armadilha do `lambdify`:** A ordem dos argumentos no `lambdify` deve ser **idêntica** à ordem que você usará na chamada da função.
    ```python
    # Se você definiu assim:
    func = lambdify((x, y), expressao, 'numpy')
    
    # Você DEVE chamar assim:
    func(valor_x, valor_y) # func(valor_y, valor_x) vai dar erro de lógica!
    ```

---

## 4. Interpretando Erros de Newton (O que o computador está dizendo?)

| Erro / Mensagem | O que significa? | Como resolver? |
| :--- | :--- | :--- |
| `LinAlgError: Singular matrix` | A Hessiana não pode ser invertida (determinante = 0). A curvatura é nula nesse ponto. | Tente um ponto inicial diferente ($X_0$). A função pode ser plana demais ali. |
| `TypeError: 'Symbol' object is not callable` | Você provavelmente esqueceu de colocar um `*` em algum lugar. Ex: `2x` em vez de `2*x`. | Verifique a sintaxe da sua função `f_sym`. |
| `Shape mismatch` | O vetor gradiente e a matriz Hessiana têm dimensões incompatíveis. | Verifique se a lista de variáveis no `hessian(f, (vars))` é a mesma do `gradiente`. |

---

## 5. Por que a Hessiana é importante para o seu projeto?
Diferente do **Gradiente Descendente** (que só vê a direção), a **Hessiana** vê a curvatura.
* Se a Hessiana for **Positiva Definida** (todos os autovalores > 0), você está em um **Mínimo Local**.
* Se o Método de Newton "explodir" (valores indo para o infinito), a função pode não ser quadrática o suficiente naquela região.



---

## 🚀 Dica Final de Performance
Se você estiver rodando o `lambdify` dentro de um loop de 1 milhão de iterações, **pare tudo!** O `lambdify` deve ser chamado **uma única vez** no início do código para criar a função. Dentro do loop, você usa apenas a função gerada por ele.

---

