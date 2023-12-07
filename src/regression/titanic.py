import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logistic as log
import linear as lin

# %%
data = np.genfromtxt(
    fname=r"C:\Users\875367\Code\jr-rsch-ml\datasets\titanic_train.csv",
    usecols=(2, 5, 6, 7, 8, 10, 12, 1),
    skip_header=1,
    delimiter=",",
    dtype=str,
)

# %%
x = data[:, :-1]
y = data[:, -1].astype(float).reshape((-1, 1))

# %%
x[:, 2:6:3][x[:, 2:6:3] == ""] = np.nan
x_nan_mean = np.nanmean(x[:, 2:6:3].astype(float))
x[:, 2:6:3] = np.nan_to_num(
    x[:, 2:6:3].astype(float),
    True,
    x_nan_mean,
)


# %%
def encode(section):
    ohe = OneHotEncoder(categories="auto")
    transformed = ohe.fit_transform(section).toarray()
    return ohe, transformed


class_ohe, class_transformed = encode(x[:, 0].reshape(-1, 1))
gender_ohe, gender_transformed = encode(x[:, 1].reshape(-1, 1))
embarked_ohe, embarked_transformed = encode(x[:, 6].reshape(-1, 1))

# %%

x = np.hstack((class_transformed, gender_transformed, x[:, 2:6], embarked_transformed))

# %%

x = x.astype(float)

# %%
x_mean = x[:, 5:9:3].mean()
x_range = x[:, 5:9:3].max() - x[:, 5:9:3].min()
x[:, 5:9:3] = (x[:, 5:9:3] - x_mean) / x_range


# %%
weights = lin.initial_weights(x)
bias = lin.add_bias(x)
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
print(f"final cost:\t\t\t{log.logistic_error(x, weights, y)}")
print(f"iterations:\t\t\t{iterations}")

# %%
p_train = x.dot(weights)
p_train = p_train > 0.5
y_train = y > 0.5
# %%
_ = log.calc_confusion_matrix(p_train, y_train, print_info=True)

for i in [0.375, 0.376, 0.377, 0.378, 0.379, 0.38, 0.381, 0.382, 0.383, 0.384, 0.385]:
    p_train = x.dot(weights)
    p_train = p_train > i
    ret = log.calc_confusion_matrix(p_train, y_train)
    print(f"{i}: {ret['accuracy']}")

# %%
# weights = array([[  1.24595163],
#        [  0.38934915],
#        [ -0.80026965],
#        [  1.76498413],
#        [ -0.92995299],
#        [-16.54471158],
#        [ -0.30368867],
#        [ -0.09313471],
#        [  1.2223888 ],
#        [  1.67872034],
#        [ -0.11663819],
#        [ -0.15248981],
#        [ -0.57456121]])

# %%
x_test = np.genfromtxt(
    fname=r"C:\Users\875367\Code\jr-rsch-ml\datasets\titanic_test.csv",
    usecols=(1, 4, 5, 6, 7, 9, 11),
    skip_header=1,
    delimiter=",",
    dtype=str,
)

# %%
x_test[:, 2:6:3][x_test[:, 2:6:3] == ""] = x_nan_mean

# %%

x_test = np.hstack(
    (
        class_ohe.fit_transform(x_test[:, 0].reshape(-1, 1)).toarray(),
        gender_ohe.fit_transform(x_test[:, 1].reshape(-1, 1)).toarray(),
        x_test[:, 2:6].astype(float),
        embarked_ohe.fit_transform(x_test[:, 6].reshape(-1, 1)).toarray(),
    )
)

# %%

x = x.astype(float)

x_test[:, 5:9:3] = (x_test[:, 5:9:3] - x_mean) / x_range

# %%
# print(log.logistic_error(x, weights, y))
# print(log.logistic_error(x_test, weights, y))
# x_test.dot(weights)[0, :]
# (x_test.dot(weights) > 0.38).astype(int)

np.savetxt(
    r"C:\users\875367\Desktop\out2.csv",
    (x_test.dot(weights) > 0.38).astype(int),
    delimiter=",",
)
