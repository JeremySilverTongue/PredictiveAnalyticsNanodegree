import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DIAMONDS_FILENAME = "diamonds.csv"
NEW_DIAMONDS_FILENAME = "new-diamonds.csv"

diamonds = pd.read_csv(DIAMONDS_FILENAME)

print diamonds.head()

X, y = diamonds[['carat', 'cut_ord', 'clarity_ord']], diamonds['price']

linear_regression = LinearRegression()
linear_regression.fit(X, y)

print linear_regression.coef_
print linear_regression.intercept_

new_diamonds = pd.read_csv(NEW_DIAMONDS_FILENAME)
new_diamonds['price'] = linear_regression.predict(new_diamonds[['carat', 'cut_ord', 'clarity_ord']])


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


fig = plt.figure(figsize=(14, 5))
ax = fig.gca()
ax.set_autoscale_on(False)
plt.xlim([0,3])
plt.ylim([-2500, 20000])
plt.scatter(rand_jitter(diamonds['carat']), rand_jitter(diamonds['price']), color="Blue", alpha=.1, s=.5)
plt.scatter(rand_jitter(new_diamonds['carat']), rand_jitter(new_diamonds['price']), color="Red", alpha=.5, s=1)

# plt.axes([0., 3., -2500., 20000.])
plt.show()
