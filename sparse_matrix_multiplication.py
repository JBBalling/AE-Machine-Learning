def sparse_matrix_multiplication(matrix_a, matrix_b):
    # Write your code here.
    if len(matrix_a[0]) != len(matrix_b):
        return [[]]

    sparse_a = get_non_zero_indices(matrix_a)
    sparse_b = get_non_zero_indices(matrix_b)
    
    matrix_c = [[0] * len(matrix_b[0]) for _ in range(len(matrix_a))]

    for i,k in sparse_a.keys():
        for j in range(len(matrix_b[0])):
            if (k,j) in sparse_b.keys():
                matrix_c[i][j] += sparse_a[(i,k)] * sparse_b[(k,j)]

    return matrix_c

def get_non_zero_indices(matrix):
    non_zero = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0: non_zero[(i, j)] = matrix[i][j]

    return non_zero
        
        
matrix_a = [
    [0, 0, 1, 2],
    [1 ,0, 1, 0]
]

matrix_b = [
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0]
]
print(sparse_matrix_multiplication(matrix_a, matrix_b))