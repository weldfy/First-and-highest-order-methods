from scipy.optimize import minimize
import numpy as np


def newton_cg(f, df, hess, x0):
    return [np.ndarray.tolist(x) for x in reversed(
            minimize(f, x0, method='Newton-CG', jac=df, options={'maxiter': 1000, 'return_all': True}, hess=hess).allvecs)]


def bfgs(f, df, hess, x0):
    return [np.ndarray.tolist(x) for x in (
            minimize(f, x0, method='BFGS', jac=df, options={'maxiter': 1000, 'return_all': True}, hess=hess).allvecs)]
