import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logistic_regression as log
import linear_regression as lin

# %%
data = np.genfromtxt(
    fname=r"C:\Users\875367\Code\jr-rsch-ml\datasets\churn_train.csv",
    usecols=(1, 2, 3, 4, 5, 6, 7, 8),
    skip_header=2,
    delimiter=",",
    dtype=str,
)

# %%
x = data[:, :-1]
# noinspection PyUnresolvedReferences
y = (data[:, -1] == "Yes").astype(float).reshape((-1, 1))


# %%
def encode(section):
    ohe = OneHotEncoder(categories="auto")
    transformed = ohe.fit_transform(section).toarray()
    return ohe, transformed


contract_ohe, contract_transformed = encode(x[:, 5].reshape(-1, 1))
method_ohe, method_transformed = encode(x[:, 6].reshape(-1, 1))

# %%
# noinspection PyUnresolvedReferences
x = np.hstack(
    (
        x[:, 0:3].astype(float),
        (x[:, 3:5] == "Yes").astype(float),
        contract_transformed,
        method_transformed,
    )
)

# %%
x_mean = x[:, 0:3].mean(axis=0)
x_std = x[:, 0:3].std(axis=0)
x[:, 0:3] = (x[:, 0:3] - x_mean) / x_std
x = lin.add_bias(x)


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
    learning_rate=1,
    precision=10,
    return_i=True,
)
print(f"final cost:\t\t\t{log.logistic_error(x, weights, y)}")
print(f"iterations:\t\t\t{iterations}")

# %%
p_train = x.dot(weights)
p_train = p_train > 0.5
y_train = y > 0.5

# %%
_ = log.calc_confusion_matrix(p_train, y_train, print_info=True)

# for i in [0.375, 0.376, 0.377, 0.378, 0.379, 0.38, 0.381, 0.382, 0.383, 0.384, 0.385]:
#     p_train = x.dot(weights)
#     p_train = p_train > i
#     ret = log.calc_confusion_matrix(p_train, y_train)
#     print(f"{i}: {ret['accuracy']}")
