MAX = 100
 
print()

def luDecomposition(mat, n):
    lower = [[0 for x in range(n)] for y in range(n)]
    upper = [[0 for x in range(n)] for y in range(n)]
    operations_count = 0  # Contador de operaciones
 
    for i in range(n):
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (lower[i][j] * upper[j][k])
                operations_count += 2  # Multiplicación y suma
 
            upper[i][k] = mat[i][k] - sum
            operations_count += 1  # Resta
 
        for k in range(i, n):
            if i == k:
                lower[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (lower[k][j] * upper[j][i])
                    operations_count += 2  # Multiplicación y suma
 
                lower[k][i] = int((mat[k][i] - sum) / upper[i][i])
                operations_count += 2  # Resta y división
 
    print("Triangular inferior\t\tTriangular superior")
 
    for i in range(n):
        for j in range(n):
            print(lower[i][j], end="\t")
        print("", end="\t")
 
        for j in range(n):
            print(upper[i][j], end="\t")
        print("")
 
    print("\nContador de operaciones:", operations_count)

    return operations_count
 
 
# factorización LU
mat = [[2, -1, -2],
       [-4, 6, 3],
       [-4, -2, 8]]
 
iter_lu = luDecomposition(mat, 3)

print()

# factorizacion QR

import numpy as np

def dot_product(vec1, vec2):
    return sum(x * y for x, y in zip(vec1, vec2))

def vector_subtraction(vec1, vec2):
    return [x - y for x, y in zip(vec1, vec2)]

def vector_scalar_multiplication(vec, scalar):
    return [x * scalar for x in vec]

def gram_schmidt(matrix):
    rows, cols = matrix.shape
    q = np.zeros((rows, cols))
    r = np.zeros((cols, cols))
    iterations = 0

    for j in range(cols):
        v = matrix[:, j]
        for i in range(j):
            r[i, j] = dot_product(q[:, i], matrix[:, j])
            v = vector_subtraction(v, vector_scalar_multiplication(q[:, i], r[i, j]))
            iterations += 2  # Cada producto punto y cada multiplicación escalar cuenta como una iteración

        r[j, j] = np.linalg.norm(v)
        q[:, j] = v / r[j, j]
        iterations += 2  # Una norma y una división cuentan como dos iteraciones adicionales

    return q, r, iterations

# matriz original
matrix1 = np.array([[2, -1, -2],
       [-4, 6, 3],
       [-4, -2, 8]])
print("Original matrix:\n", matrix1)

# descomposicion qr
q, r, iterations = gram_schmidt(matrix1)
print('\nQ:\n', q)
print('\nR:\n', r)
print('\nContador de operaciones:', iterations)

print()

if iter_lu < iterations:
    print(f"LU es más eficiente que QR ({iter_lu} iteraciones para LU < {iterations} iteraciones para QR)")
elif iter_lu > iterations:
    print(f"QR es más eficiente que LU ({iter_lu} iteraciones para LU > {iterations} iteraciones para QR)")
else:
    print("Ambos métodos son igual de eficientes")

print()