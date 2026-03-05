import sympy as sp

# 1. Declarando as Variáveis Simbólicas
# Isso diz ao Python: "Não trate x e y como números, trate como símbolos de álgebra"
x, y = sp.symbols("x y")

# 2. Definindo a Função de Custo
f = x**2 + 2 * y**2

print(f"Função Original: f(x,y) = {f}")

# 3. Calculando as Derivadas Parciais (O método .diff)
# sp.diff(função, variável_alvo)
df_dx = sp.diff(f, x)  # Derivada em relação a x (congelando y)
df_dy = sp.diff(f, y)  # Derivada em relação a y (congelando x)

print(f"Derivada parcial em x (df/dx): {df_dx}")
print(f"Derivada parcial em y (df/dy): {df_dy}")

# 4. Montando o Vetor Gradiente Simbólico
gradiente = [df_dx, df_dy]
print(f"Vetor Gradiente Simbólico: {gradiente}")

# 5. Avaliando o Gradiente em um Ponto Específico: P(4, 3)
# Usamos o método .subs() para substituir os símbolos por valores numéricos
ponto = {x: 4, y: 3}

grad_x_num = df_dx.subs(ponto)
grad_y_num = df_dy.subs(ponto)

print(f"\n--- Resultado Final ---")
print(f"Vetor Gradiente no ponto (4,3): [{grad_x_num}, {grad_y_num}]")
