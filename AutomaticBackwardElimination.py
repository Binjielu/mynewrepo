import statsmodels.api as sm
import numpy as np


class BackwardElimination:
    def __init__(self, x, y, sl=0.05):
        self.y = y
        self.sl = sl

    def backward_elimination(self, x):
        num_vars = len(x[0])
        for i in range(num_vars):
            regressor_ols = sm.OLS(self.y, x).fit()
            max_var = max(regressor_ols.pvalues).astype(float)
            if max_var > self.sl:
                for j in range(num_vars - i):
                    if regressor_ols.pvalues[j].astype(float) == max_var:
                        x = np.delete(x, j, 1)
        regressor_ols.summary()
        return x

    def backward_elimination_two(self, x):
        num_vars = len(x[0])
        temp = np.zeros(x.shape).astype(int)
        for i in range(num_vars):
            regressor_ols = sm.OLS(self.y, x).fit()
            max_var = max(regressor_ols.pvalues).astype(float)
            adj_r_before = regressor_ols.rsquared_adj.astype(float)
            if max_var > self.sl:
                for j in range(num_vars - i):
                    if regressor_ols.pvalues[j].astype(float) == max_var:
                        temp[:, j] = x[:, j]
                        x = np.delete(x, j, 1)
                        tmp_regressor = sm.OLS(self.y, x).fit()
                        adj_r_after = tmp_regressor.rsquared_adj.astype(float)
                        if adj_r_before >= adj_r_after:
                            x_rollback = np.hstack((x, temp[:, [0, j]]))
                            x_rollback = np.delete(x_rollback, j, 1)
                            print(regressor_ols.summary())
                            return x_rollback
                        else:
                            continue
        regressor_ols.summary()
        return x
