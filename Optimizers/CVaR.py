import cvxpy as cp
import numpy as np

### Optimizers
def cvar_optimization(data, alpha=0.05):
    """
    This function optimizes the CVaR for a given portfolio.

    :param data: pd.DataFrame: a pandas dataframe where each column represents a security and each row represents a time.
    :param alpha: float: the confidence level used to compute the VaR and CVaR. Default is 0.05.
    :return: np.array: the optimal weights for the portfolio.
    """
    n = data.shape[1]
    returns = data.pct_change().dropna().values

    w = cp.Variable(n)
    gamma = cp.Variable()
    z = cp.Variable(returns.shape[0])

    objective = cp.Minimize(gamma + 1.0/alpha * cp.sum(z)/returns.shape[0])
    constraints = [cp.sum(w) == 1, 
                   w >= 0, 
                   returns @ w - gamma <= z, 
                   z >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return w.value
