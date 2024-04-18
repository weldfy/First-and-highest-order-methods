import numpy as np


def golden_ratio(func, a, b, tol=1e-6):
    gr = (1 + np.sqrt(5)) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if func(c) < func(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2


def newton_method_with_line_search(func, grad_func, hess_func, x, tol=1e-6, max_iter=1000):
    res = [x]
    for _ in range(max_iter):
        grad = grad_func(x)
        hess = hess_func(x)
        step = np.linalg.solve(hess, grad)
        alpha = golden_ratio(lambda a: func(x - a * step), 0, 1)
        x = x - alpha * step
        res.append(np.ndarray.tolist(x))
        if np.linalg.norm(step) < tol:
            break
    return res
