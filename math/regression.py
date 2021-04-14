import numpy as np
import least_squares as lsq

def fit_polynom(xs, ys, degree = 1, ws = 1, xs_test = None):
    degree += 1
    state = lsq.initialize(degree)
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        
        vals = np.empty(degree)
        for j in range(degree):
            vals[j] = x**j
        if ws is None:
            w = 1.
        elif isinstance(ws, (list, tuple, np.ndarray)):
            w = ws[i]
        else:
            w = ws
        state = lsq.add_measurement(vals, y, w, state)
    coefs = lsq.solve(lsq.invert(state))
    if xs_test is not None:
        powers = np.arange(len(coefs))
        powers = np.reshape(np.repeat(powers, len(xs_test)), (len(powers), len(xs_test)))
        ws = np.reshape(np.repeat(coefs, len(xs_test)), (len(powers), len(xs_test)))
        ys = np.sum(ws*xs_test**powers, axis=0)
        return coefs, ys
    else:
        return coefs

def fit_periodic_polynom(xs, ys, period, degree = 2, ws = 1., smoothness_level = 1, xs_test = None):
    if degree < smoothness_level:
        raise ValueError("degree should be >= {smoothness_level} for smoothness_level = {smoothness_level}" )
    if smoothness_level > 2:
        raise ValueError("Unsupported smoothness_level" )
    degree += 1
    state = lsq.initialize(2*degree)
    for k in [0, 1]:
        for i in range(len(xs)):
            x = xs[i] + k*period
            y = ys[i]
            
            vals = np.zeros(2*degree)
            for j in range(0, degree):
                vals[j+k*degree] = x**j
            if ws is None:
                w = 1.
            elif isinstance(ws, (list, tuple, np.ndarray)):
                w = ws[i]
            else:
                w = ws
            state = lsq.add_measurement(vals, y, w, state)
    state = lsq.invert(state)

    vals = np.zeros(2*degree)
    for j in range(0, degree):
        vals[j] = period**j
        vals[j + degree] = -vals[j]
    state = lsq.add_constraint(vals, 0, 0, state)
    for j in range(0, degree):
        vals[j] = (period/2)**j
        vals[j + degree] = -(period + period/2)**j
    state = lsq.add_constraint(vals, 0, 0, state)

    if smoothness_level > 0:
        vals = np.zeros(2*degree)
        for j in range(0, degree):
            vals[j] = j*period**(j-1)
            vals[j + degree] = -vals[j]
        state = lsq.add_constraint(vals, 0, 0, state)
        vals = np.zeros(2*degree)
        for j in range(0, degree):
            vals[j] = j*(period/2)**(j-1)
            vals[j + degree] = -j*(period + period/2)**(j-1)
        state = lsq.add_constraint(vals, 0, 0, state)

    if smoothness_level > 1:
        vals = np.zeros(2*degree)
        for j in range(0, degree):
            vals[j] = j*(j-1)*period**(j-2)
            print(j, vals[j])
            vals[j + degree] = -vals[j]
        state = lsq.add_constraint(vals, 0, 0, state)
        vals = np.zeros(2*degree)
        for j in range(0, degree):
            vals[j] = j*(j-1)*(period/2)**(j-2)
            vals[j + degree] = -j*(j-1)*(period + period/2)**(j-2)
        state = lsq.add_constraint(vals, 0, 0, state)
        
        
    coefs = lsq.solve(state)
    if xs_test is not None:
        coefs1 = coefs[:-degree]
        coefs2 = coefs[degree:]
        powers1 = np.arange(len(coefs1))
        powers1 = np.reshape(np.repeat(powers1, len(xs_test)), (len(powers1), len(xs_test)))
        ws1 = np.reshape(np.repeat(coefs1, len(xs_test)), (len(powers1), len(xs_test)))
        ys1 = np.sum(ws1*xs_test**powers1, axis=0)

        powers2 = np.arange(len(coefs2))
        powers2 = np.reshape(np.repeat(powers2, len(xs_test)), (len(powers2), len(xs_test)))
        ws2 = np.reshape(np.repeat(coefs2, len(xs_test)), (len(powers2), len(xs_test)))
        ys2 = np.sum(ws2*(xs_test+period)**powers2, axis=0)
        ys = np.concatenate([ys2[:len(ys2)//2], ys1[len(ys1)//2:]])
        return coefs, ys
    else:
        return coefs


if (__name__ == '__main__'):
    import sys
    import os
    sys.path.append('..')
    sys.path.append('../utils')
    import plot
    
    period = 4
    coefs = [.4, -.3, .2, -.05]
    xs = np.linspace(0, period, 20)
    powers = np.arange(len(coefs))
    powers = np.reshape(np.repeat(powers, len(xs)), (len(powers), len(xs)))
    coefs = np.reshape(np.repeat(coefs, len(xs)), (len(powers), len(xs)))
    ys = np.sin(2*np.pi*xs/period) + .1*np.random.normal(size=len(xs))
    
    fig = plot.plot()
    fig.plot(xs, ys, "b+")
    
    xs_test=np.linspace(0, period, 1000)
    coefs, ys_test = fit_periodic_polynom(xs, ys, period=period, degree=3, xs_test=xs_test)
    fig.plot(xs_test, ys_test, "k-")
    print(coefs)
    print(len(ys_test))
    print(ys_test[0] - ys_test[-1])
    print(ys_test[499]-ys_test[500])
    #print(ys_test[999]-ys_test[1000])
    #print(ys_test[250]-ys_test[1250])
    #print(.25-xs_test[24]/period, xs_test[25]/period-.25)
    fig.save("polynom_fit.png")
