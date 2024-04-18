import numpy as np


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                     200 * (x[1] - x[0] ** 2)])


def rosenbrock_hess(x):
    return np.array([[2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]],
                     [-400 * x[0], 200]])


def quadratic_function(x):
    return 3 * x[0] ** 2 + 5 * x[1] ** 2 + 2 * x[0] * x[1]


def quadratic_grad(x):
    return np.array([6 * x[0] + 2 * x[1], 10 * x[1] + 2 * x[0]])


def quadratic_hess(x):
    return np.array([[6, 2],
                     [2, 10]])


def quadratic_function_2(x):
    return 2 * x[0] ** 2 - 4 * x[0] * x[1] + 3 * x[1] ** 2


def quadratic_grad_2(x):
    return np.array([4 * x[0] - 4 * x[1], -4 * x[0] + 6 * x[1]])


def quadratic_hess_2(x):
    return np.array([[4, -4],
                     [-4, 6]])
