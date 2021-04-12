import numpy as np

def calc_forward(i, m):
    p = len(m) - 1
    col = m[:, i]
    i_max = i + np.argmax(np.abs(col[i:]))
    if i != i_max:
        m = np.concatenate([m[:i, :], m[i_max, :][None,:], m[i+1:i_max], m[i, :][None,:], m[i_max+1:,:]], axis=0)
        
    aii = 1. / m[i, i]
    for j in range(i + 1, p + 1):
        aji = m[j, i]
        for k in range(i + 1, p + 2):
            m[j, k] = m[j, k] - aji * aii * m[i, k]
    return m

def back_substitution(m):
    num_rows = len(m)
    ret_val = np.zeros(num_rows)
    p = num_rows - 1

    for i in range(p, -1, -1):
        #print(i, m)
        sum_ = 0
        for j in range(i + 1, p + 1):
            sum_ +=  m[i, j] * ret_val[j]
        ret_val[i] = (m[i, (p + 1)] - sum_) / m[i, i]
    return ret_val

def solve_gauss(m):
    for i in range(0, len(m) - 1):
        m = calc_forward(i, m)
    print(m)
    return back_substitution(m)
    