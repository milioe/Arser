import numpy as np
from scipy.optimize import minimize

def MV_criterion(weights, data):
    Lambda = 1
    W = 1
    Wbar = 1 + 0.25 / 100
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return.sum(axis=1)
    mean = np.mean(portfolio_return, axis=0)
    std = np.std(portfolio_return, axis=0)
    criterion = Wbar ** (1 - Lambda) / (1 + Lambda) + Wbar ** (-Lambda) \
                * W * mean - Lambda / 2 * Wbar ** (-1 - Lambda) * W ** 2 * std ** 2
    criterion = -criterion
    return criterion


def obtain_mv(database, train_set):
    n = database.shape[1]
    x0 = np.ones(n)
    cons = ({'type': 'eq', 'fun': lambda x: sum(abs(x)) - 1})
    Bounds = [(0, 1) for i in range(0, n)]
    res_MV = minimize(MV_criterion, x0, method="SLSQP", args=(train_set), bounds=Bounds, constraints=cons, options={'disp': True})
    X_MV = res_MV.x
    return X_MV