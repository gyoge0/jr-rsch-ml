import numpy as np
from sklearn.preprocessing import OneHotEncoder
from logistic import logistic_error, gradient
import logistic as log
import linear as lin

# %%
data = np.genfromtxt(
    fname=r"C:\Users\875367\Code\jr-rsch-ml\datasets\linear\iris.csv",
    usecols=(0, 1, 2, 3, 4),
    skip_header=1,
    delimiter=",",
    dtype=str,
)

# %%
x = data[:, :-1].astype(float)
y = data[:, -1].reshape((-1, 1))

# %%
ohe = OneHotEncoder(categories="auto")
y = ohe.fit_transform(y).toarray()

# %%
x_mean = x.mean()
x_range = x.max() - x.min()
x = (x - x_mean) / x_range

# %%
x = lin.add_bias(x)
weights = log.initial_weights(x, num=3)
print(f"initial cost:\t\t\t{log.logistic_error(x, weights, y)}")

# %%
weights, iterations = log.fit_regression(
    x,
    y,
    weights,
    print_info=False,
    check_interval=1000,
    learning_rate=0.8,
    precision=4,
    return_i=True,
)

# %%
# weights = []
# for i in range(3):
#     weights.append(log.initial_weights(x, num=1))
#     weights_new, iterations = log.fit_regression(
#         x,
#         y[:, i].reshape((-1, 1)),
#         weights[i],
#         print_info=False,
#         check_interval=1000,
#         learning_rate=0.8,
#         precision=4,
#         return_i=True,
#     )
#     weights[i] = weights_new
# weights = np.hstack(tuple(weights))

# %%
predictions = log.argmax_predict(x, weights).reshape((-1, 1)) + 1
actual = data[:, -1].astype(int).reshape((-1, 1))
correct = predictions == actual
correct_count = np.sum(correct)

# %%
# np.hstack((log.argmax_predict(x, weights).reshape((-1, 1)) + 1, data[:, -1].astype(int).reshape((-1, 1))))
