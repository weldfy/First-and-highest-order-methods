import numpy as np


def armijo_condition(f, f_prime, xk, pk, alpha, c1):
    return f(xk + alpha * pk) <= f(xk) + c1 * alpha * np.dot(f_prime(xk), pk)


def curvature_condition(f, f_prime, xk, pk, alpha, c2):
    return np.dot(f_prime(xk + alpha * pk), pk) >= c2 * np.dot(f_prime(xk), pk)


def wolfe_search(f, f_prime, xk, pk, c1=1e-4, c2=0.9, alpha0=1.0, max_iter=1000):
    alpha = alpha0
    for _ in range(max_iter):
        if armijo_condition(f, f_prime, xk, pk, alpha, c1) and curvature_condition(f, f_prime, xk, pk, alpha, c2):
            return alpha
        alpha *= 0.5
    return alpha


def newton_method(f, f_prime, f_hessian, x, tol=1e-6, max_iter=1000):
    res = [x]
    for _ in range(max_iter):
        grad = f_prime(x)
        hessian = f_hessian(x)
        pk = -np.linalg.solve(hessian, grad)
        alpha = wolfe_search(f, f_prime, x, pk)
        x_new = x + alpha * pk
        res.append(np.ndarray.tolist(x_new))
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return res
