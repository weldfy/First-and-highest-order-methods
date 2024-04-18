import numpy as np


def newton_method(func, grad_func, hess_func, x, h=1, tol=1e-6, max_iter=1000):
    res = [x]
    for _ in range(max_iter):
        grad = grad_func(x)
        hess = hess_func(x)
        step = np.linalg.solve(hess, grad)
        x = x - step * h
        res.append(np.ndarray.tolist(x))
        if np.linalg.norm(step) < tol:
            break
    return res
