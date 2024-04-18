import numpy as np


def armijo_condition(f, grad, xk, pk, alpha, c1):
    return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(grad(xk), pk)


def curvature_condition(f, grad, xk, pk, alpha, c2):
    return np.dot(grad(xk + alpha * pk), pk) >= c2 * np.dot(grad(xk), pk)


def wolfe_search(f, grad, xk, pk, c1, c2, alpha0=1.0, max_iter=1000):
    alpha = alpha0
    for _ in range(max_iter):
        if armijo_condition(f, grad, xk, pk, alpha, c1) and curvature_condition(f, grad, xk, pk, alpha, c2):
            return alpha
        alpha *= 0.5
    return alpha


def newton_method(f, grad, hess, x, c1=1e-4, c2=0.9, tol=1e-6, max_iter=1000):
    res = [x]
    for _ in range(max_iter):
        grad = grad(x)
        pk = -np.linalg.solve(hess(x), grad)
        alpha = wolfe_search(f, grad, x, pk, c1, c2)
        x_new = x + alpha * pk
        res.append(np.ndarray.tolist(x_new))
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return res
