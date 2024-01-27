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

# Original matrix
matrix1 = np.array([[2, -1, -2],
       [-4, 6, 3],
       [-4, -2, 8]])
print("Original matrix:\n", matrix1)

# Decomposition of the matrix
q, r, iterations = gram_schmidt(matrix1)
print('\nQ:\n', q)
print('\nR:\n', r)
print('\nNumber of iterations:', iterations)
