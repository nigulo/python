import numpy as np
import linear_equations as le

def initialize(p):
    '''
    p - Number of x variables.
    Returns the initial state for weighted least square computation.
    State is a tuple (b, d, e), where b is the unit upper triangular matrix and
    d is d or f depending on if Gentleman's or tirlings procedure is used.
    '''
    b = np.zeros((p, p + 1))
    d = np.zeros(p + 1)
    for j in range(0, p):
        b[j, j] = 1.
    return (b, d, 0)

def add_measurement(x, y, w, state):
    '''
    Adds a next row of measurements to the computation. If you have
    finished adding measurements, call 'invert' and start adding constraints 
    using 'add_constraint'. In case there are no constraints to impose
    immediately call solve to obtain the LSQ estimate.

    x - array of x values.
    y - y value.
    w - weight of the measurement (0 = don't use).
    state - previous state of calculation.
    Returns next state of calculation
    '''
    if w == 0:
        return state
    
    x = np.concatenate([x, [y]])

    (b, d, e) = state
    p = len(x) - 1
    if p != len(b):
        raise ValueError("Invalid input vector")

    d[p] = w
    for i in range(0, p):
        h = x[i]
        dm = d[p]
        if h != 0 and dm != 0:
            di = d[i]
            d_ =  di + dm * h * h
            c_ = di / d_
            s_ = dm * h / d_
            d[i] = d_
            d[p] = c_ * dm
            
            for k in range(i + 1, p + 1):
                fkb = b[i, k]
                gkx = x[k]
                b[i, k] = c_ * fkb + s_ * gkx
                x[k] = gkx - h * fkb

    xyp = x[p]
    coef = d[p]
    return (b, d, e + coef * xyp * xyp)


def invert(state):
    '''
    state - previous state of calculation.
    Returns next state of calculation
    '''
    (b, d, e) = state
    d = 1./d
    return (b, d, e)


def add_constraint(r, s, w, state):
    '''
    Adds a next row of constraints to the computation. If you have
    finished adding constraints, call 'solve' to obtain the LSQ estimate.
    r - array of r values.
    s - s value.
    w - weight of the constraint (0 = infinite).
    state - previous state of calculation.
    Returns next state of calculation
    
    '''
    r = np.concatenate([r, [s]])
    (b, f, e) = state
    p = len(r) - 1
    if p != len(b):
        raise ValueError("Invalid input vector")

    f[p] = w
    for i in range(0, p):
        h = r[i]
        fi = f[i]
        if fi == 0 and h != 0:
            for k in range(i + 1, p + 1):
                r[k] = r[k] - h * b[i, k]
        elif h != 0:
            fm = f[p]
            f_ =  fm + fi * h * h
            c_ = fm / f_
            s_ = fi * h / f_
            f[i] = c_ * fi
            f[p] = f_
            for k in range(i + 1, p + 1):
                g1 = b[i, k]
                g2 = r[k]
                b[i, k] = c_ * g1 + s_ * g2
                r[k] = g2 - h * g1

    rsp = r[p]
    fm = f[p]
    if fm != 0:
        e1 = e + rsp * rsp / fm
    else:
        e1 = e
    return (b, f, e1)

def solve(state):
    (b, _, _) = state
    return le.back_substitution(b)
