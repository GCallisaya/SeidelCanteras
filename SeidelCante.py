def gauss_seidel(A, b, x0=None, tol=1e-10, max_iterations=100):
    n = len(b)
    x = [0.0] * n if x0 is None else x0.copy()
    
    for it_count in range(max_iterations):
        x_old = x.copy()
        
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]
        
        # Cálculo del error
        error = max(abs(x[i] - x_old[i]) for i in range(n))
        print(f"Iteración {it_count + 1}: x = {x}, error = {error}")

        if error < tol:
            print(f"Convergió en {it_count + 1} iteraciones.")
            return x
    
    print("No convergió.")
    return x

# Ejemplo de uso:
A = [[52, 20, 25],
     [30, 50, 20],
     [18, 30, 55]
    ]

b = [4800, 5810, 5690]

# Llamada a la función
x0 = [0.0] * len(b)  # Valor inicial
solucion = gauss_seidel(A, b, x0)

print("Solución final:", solucion)
