# 📝 Lista de Aquecimento: Dominando Derivadas Parciais

**Disciplina:** ICD304 – Métodos de Otimização para IA e CD

**Objetivo:** Praticar a "Regra do Congelamento" para o cálculo de gradientes.

**Instruções:** Calcule as derivadas parciais para cada uma das funções abaixo. Lembre-se do apesentado em aula: quando derivamos em relação a uma variável, **todas as outras são tratadas como números constantes**.

---

### 🟢 Exercício 1: O Básico (Soma de Termos)

Calcule $\frac{\partial f}{\partial x}$ e $\frac{\partial f}{\partial y}$ para a função:

$$f(x, y) = 4x^3 + 7y^2 - 5$$


### 🟡 Exercício 2: O Produto

Calcule $\frac{\partial f}{\partial x}$ e $\frac{\partial f}{\partial y}$ para a função:

$$f(x, y) = 5x^2y^3$$

**Dica:** Aqui a função não é uma soma entre os termos independentes, mas uma multiplicação. Veja a regra da cadeia para esse caso.

---

### 🟠 Exercício 3: Subindo de Dimensão (Três Variáveis)

Calcule o vetor gradiente $\nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right]$ para:

$$f(x, y, z) = x^2yz^3 - 4xy$$

---

### 🔴 Exercício 4: Funções Exponenciais (Preparação para IA)

Calcule $\frac{\partial f}{\partial x}$ e $\frac{\partial f}{\partial y}$ para a função:$$f(x, y) = e^{3x}y^2$$

---

### 🟣 Exercício 5: O "Chefe Final" (O Erro Quadrático de um Neurônio)

Em Machine Learning, queremos ajustar o peso ($w$) e o viés ($b$) para minimizar uma função de perda ($L$). 
Imagine que a função de perda para um único dado seja:

$$L(w, b) = (2w + b - 10)^2$$

(Dica: Use a Regra da Cadeia do Cálculo 1: derive "por fora" e multiplique pela derivada "de dentro"). 

Calcule $\frac{\partial L}{\partial w}$ e $\frac{\partial L}{\partial b}$.


---
