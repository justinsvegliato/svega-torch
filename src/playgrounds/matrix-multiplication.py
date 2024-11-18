import torch
import time


def matrix_multiply_1(a, b):
    assert a.shape[1] == b.shape[0]

    num_rows = a.shape[0]
    num_cols = b.shape[1]

    results = torch.zeros(num_rows, num_cols)

    for row_index in range(num_rows):
        for col_index in range(num_cols):
            for element_index in range(num_rows):
                results[row_index, col_index] += a[row_index, element_index] * b[element_index, col_index]

    return results


def matrix_multiply_2(a, b):
    assert a.shape[1] == b.shape[0]

    num_rows = a.shape[0]
    num_cols = b.shape[1]

    results = torch.zeros(num_rows, num_cols)

    for row_index in range(num_rows):
        for col_index in range(num_cols):
            results[row_index, col_index] += (a[row_index,:] * b[:,col_index]).sum()

    return results


def matrix_multiply_3(a, b):
    assert a.shape[1] == b.shape[0]

    num_rows = a.shape[0]
    num_cols = b.shape[1]

    results = torch.zeros(num_rows, num_cols)

    for row_index in range(num_rows):
        row = a[row_index].unsqueeze(-1)
        results[row_index] = (row * b).sum(dim=0)

    return results


def matrix_multiply_4(a, b):
    return torch.einsum("ab,bc->ac", a, b)


def matrix_multiply_5(a, b):
    return a @ b


def main():
    a = torch.ones(100, 200)
    b = torch.ones(200, 100)

    start_time = time.time()
    m1 = matrix_multiply_1(a, b)
    print(m1)
    print("matrix_multiply_1 took: {:.6f} seconds".format(time.time() - start_time))
    
    start_time = time.time()
    m2 = matrix_multiply_2(a, b)
    print(m2)
    print("matrix_multiply_2 took: {:.6f} seconds".format(time.time() - start_time))
    
    start_time = time.time()
    m3 = matrix_multiply_3(a, b)
    print(m3)
    print("matrix_multiply_3 took: {:.6f} seconds".format(time.time() - start_time))
    
    start_time = time.time()
    m4 = matrix_multiply_4(a, b)
    print(m4)
    print("matrix_multiply_4 took: {:.6f} seconds".format(time.time() - start_time))

    start_time = time.time()
    m5 = matrix_multiply_5(a, b)
    print(m5)
    print("matrix_multiply_5 took: {:.6f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
