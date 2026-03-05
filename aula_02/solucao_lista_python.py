import sympy as sp

x = sp.symbols('x')

f = x**3
g = 5
h = x**2 + 3*x
## uso de sp.sin - símbolo específico da biblioteca
i = x**2 * sp.sin(x)
j = x / (x+1)
## uso de sp.sin - símbolo específico da biblioteca
k = sp.sin(x**2)

f_deriv = sp.diff(f,x)
g_deriv = sp.diff(g,x)
h_deriv = sp.diff(h,x)
i_deriv = sp.diff(i,x)
j_deriv = sp.diff(j,x)
k_deriv = sp.diff(k,x)

print(f"f'(x) = {f_deriv}")
print(f"g'(x) = {g_deriv}")
print(f"h'(x) = {h_deriv}")
print(f"i'(x) = {i_deriv}")
print(f"j'(x) = {j_deriv}")
print(f"k'(x) = {k_deriv}")


