from scipy.optimize import minimize
import numpy as np


def newton_cg(f, df, hess, x0):
    return [np.ndarray.tolist(x) for x in reversed(
            minimize(f, x0, method='Newton-CG', jac=df, options={'maxiter': 1000, 'return_all': True}).allvecs)]


def bfgs(f, df, hess, x0, c1=0.0001, c2=0.9):
    return [np.ndarray.tolist(x) for x in (
            minimize(f, x0, method='BFGS', jac=df, options={'maxiter': 1000, 'return_all': True, 'c1': c1, 'c2': c2}).allvecs)]
