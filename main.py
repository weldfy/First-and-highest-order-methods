import random

import functions
import newton_method
import newton_method_with_line_search
import scipy_methods
import wolfe_newton_method
import visualizer

methods = [
    {
        'run': newton_method.newton_method,
        'name': 'Newton',
    },
    {
        'run': newton_method_with_line_search.newton_method_with_line_search,
        'name': 'Newton with golden ratio search',
    },
    {
        'run': scipy_methods.newton_cg,
        'name': 'Newton-CG',
    },
    {
        'run': scipy_methods.bfgs,
        'name': 'BFGS',
    },
    {
        'run': wolfe_newton_method.newton_method,
        'name': 'wolfe',
    }
]

funcs = [
    {
        'func': functions.rosenbrock,
        'grad': functions.rosenbrock_grad,
        'hess': functions.rosenbrock_hess,
    },
    {
        'func': functions.quadratic_function,
        'grad': functions.quadratic_grad,
        'hess': functions.quadratic_hess,
    },
    {
        'func': functions.quadratic_function_2,
        'grad': functions.quadratic_grad_2,
        'hess': functions.quadratic_hess_2,
    }
]

if __name__ == "__main__":
    x = [random.randrange(-100, 100), random.randrange(-100, 100)]
    f = funcs[0]
    method = methods[4]
    print("start point:", x)
    result = method['run'](f['func'], f['grad'], f['hess'], x)
    print('minimum point:', result[-1])
    print('minimum result:', f['func'](result[-1]))
    print('operations count:', len(result))
    visualizer.visual_optimization3d(f['func'], result,
                                     method['name'], [-200, 200], [-200, 200])

