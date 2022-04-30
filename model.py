import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('https://github.com/mostafa-zidane/mlproj/raw/main/finaldata.csv')

y = df['Premium']
X = df.loc[:, df.columns != 'Premium']

class Ridge_():
    def __init__(self, alpha):
        self.betas = None
        self.columns = None
        self.alpha = alpha

    def removecorr(self, X):
        Xx = X.copy()
        cor = Xx.corr()
        cor1 = cor.values
        np.fill_diagonal(cor1, 0)
        col = pd.DataFrame(cor1, columns=cor.columns, index=cor.columns)
        col = col[(col == 1).any(axis=1)]  # selecting only rows which have correlation coefficient 1.0
        # this loop runs while there are still variables in the data that are collinear
        while (not col.empty):
            # storing in "columnnames" the names of the variables that are perfectly correlated with the first row (first variable) in col
            columnnames = col.loc[col.index[0], col.iloc[0] == 1].index
            # dropping these columns from the data
            Xx.drop(list(columnnames), axis=1, inplace=True)
            # dropping the first row and the rows of the variables it's correlated with from col (collinearity is resolved)
            col.drop(col.index[0], axis=0, inplace=True)
            col.drop(list(columnnames), axis=0, inplace=True)

        return Xx

    def fit(self, X, y):
        Xx = X.copy()
        Xx = self.removecorr(Xx)
        Xx['ones'] = 1
        am = np.identity(len((Xx.T @ Xx)), dtype=float)
        self.columns = list(Xx.columns)
        self.betas = np.linalg.solve(((Xx.T @ Xx) + np.dot(am, self.alpha)), (Xx.T @ y))

    def predict(self, X_test):
        Xtest = X_test.copy()
        Xtest['ones'] = 1
        Xtest = Xtest[self.columns]

        return np.dot(Xtest, self.betas)


class Ridge_PolynomialRegression():
    def __init__(self, degree, alpha):
        self.linreg = Ridge_(alpha)
        if (degree < 2):
            print('degree must be greater than one')
            degree = 2
        self.degree = degree

    def poly_transform(self, X):
        pol1 = X.copy()
        pol = X.copy()
        for i in range(2, self.degree + 1):
            pol = pol.join(pol1 ** i, how='left', rsuffix='_' + str(i))
        return pol

    def fit(self, x, y):
        Xx = self.poly_transform(x)
        self.linreg.fit(Xx, y)

    def predict(self, x):
        Xx = self.poly_transform(x)
        return self.linreg.predict(Xx)

regressor = Ridge_PolynomialRegression(3, 0.053)
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('regressor.pkl', 'wb'))