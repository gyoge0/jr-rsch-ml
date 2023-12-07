import numpy as np
from sklearn.preprocessing import OneHotEncoder
import regression.logistic as log
import regression.linear as lin

# %%
data = np.genfromtxt(
    fname=r"C:\Users\875367\Code\jr-rsch-ml\datasets\1b_train.csv",
    usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    skip_header=1,
    delimiter=",",
    dtype=str,
)

# %%
x = data[:, :-1]
# noinspection PyUnresolvedReferences
y = data[:, -1].astype(float).reshape((-1, 1))
x_before = x.copy()


# %%
def encode(section):
    ohe = OneHotEncoder(categories="auto")
    transformed = ohe.fit_transform(section).toarray()
    return ohe, transformed


glucose_ohe, glucose_transformed = encode(x[:, 8].reshape(-1, 1))
cholesterol_ohe, cholesterol_transformed = encode(x[:, 9].reshape(-1, 1))

# %%
# noinspection PyUnresolvedReferences
x = np.hstack(
    (
        x[:, 0:4].astype(float),
        (x[:, 4].astype(float) - 1).reshape(-1, 1),
        x[:, 5:8].astype(float),
        glucose_transformed,
        cholesterol_transformed,
    )
)

# %%
x_mean = x[:, 0:4].mean(axis=0)
x_std = x[:, 0:4].std(axis=0)
x[:, 0:4] = (x[:, 0:4] - x_mean) / x_std

# %%
bias = np.ones((len(x), 1))
x = np.hstack((bias, x))

# %%
weights = lin.initial_weights(x)
print(f"initial cost:\t\t\t{log.logistic_error(x, weights, y)}")

# %%
weights, iterations = log.fit_regression(
    x,
    y,
    weights,
    print_info=True,
    check_interval=100,
    learning_rate=0.1,
    precision=10,
    return_i=True,
)
print(f"final cost:\t\t\t{log.logistic_error(x, weights, y)}")
print(f"iterations:\t\t\t{iterations}")

# %%
p_train = log.sigmoid(x.dot(weights))
p_train = p_train > 0.5
y_train = y > 0.5

# %%
matrix = log.calc_confusion_matrix(p_train, y_train, print_info=True)

# for i in [0.375, 0.376, 0.377, 0.378, 0.379, 0.38, 0.381, 0.382, 0.383, 0.384, 0.385]:
#     p_train = x.dot(weights)
#     p_train = p_train > i
#     ret = log.calc_confusion_matrix(p_train, y_train)
#     print(f"{i}: {ret['accuracy']}")
