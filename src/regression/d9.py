import numpy as np
import matplotlib.pyplot as plt
import linear as lin


def seperator():
    print("-" * 8)


# %%
f_name = "2D"
skip_header = True
# use_cols = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
# use_cols = (0, 1, 2, 6)
# use_cols = (2, 3)
# use_cols = (0, 1, 2, 4)
# use_cols = (1, 0)
use_cols = (0, 1, 2, 3, 4)

# %%

# load train data
ds = np.genfromtxt(
    fname=rf"C:\Users\875367\Code\jr-rsch-ml\datasets\{f_name}_train.csv",
    delimiter=",",
    dtype=np.float64,
    skip_header=skip_header,
    usecols=use_cols,
)

# %%
with_bias = lin.add_bias(ds)

seperator()
print(f"example: {with_bias[0, :]}")

# %%
x_train, y_train = lin.create_xy(ds)

seperator()
print(f"example standardized: {x_train[0, :]}")
# %%
_, mean, std = lin.normalize_x(ds[:, :-1], ds)

seperator()
print(f"mean: {mean}")
print(f"std: {std}")


# %%
r, c = x_train.shape
w_initial = lin.initial_weights(x_train)

seperator()
print(f"initial weights: {w_initial}")

# %%
w_final, ret = lin.fit_regression(
    x=x_train,
    y=y_train,
    weights=w_initial,
    max_iterations=1000000,
    learning_rate=0.7,
    goal_error=0.00001,
    precision=7,
    print_info=True,
    check_interval=1,
    return_i=False,
    ret=True,
)

seperator()
print(f"final weights: {w_final}")
print(f"initial cost: {lin.mean_square_error(x_train, w_initial, y_train)}")
print(f"final cost: {lin.mean_square_error(x_train, w_final, y_train)}")

# %%
w_fancy = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
print(f": fancy weights: {w_fancy}")

# %%

best_lr, best_i = lin.best_lr(
    x=x_train,
    y=y_train,
    w_initial=w_initial,
    lower_bound=0.1,
    upper_bound=1.5,
    step=0.05,
)
print(f"best learning rate: {best_lr}")
print(f"iterations: {best_i}")

# %%
# load test data
ds_test = np.genfromtxt(
    fname=rf"C:\Users\875367\Code\jr-rsch-ml\datasets\{f_name}_test.csv",
    delimiter=",",
    dtype=np.float64,
    skip_header=True,
    usecols=use_cols,
)

with_bias_test = lin.add_bias(ds_test)

seperator()
print(f"example test: {with_bias_test[:3, :]}")

# %%
x_test = with_bias_test[:, :-1]
x_test[:, 1:] = (x_test[:, 1:] - mean) / std
y_test = ds_test[:, -1][:, None]

seperator()
print(f"example test standardized: {x_test[:3, :]}")

# %%
test_prediction = x_test.dot(w_final)
test_diff = test_prediction - y_test

seperator()
print(f"prediction: {test_prediction}")
print(f"actual: {y_test}")
print(f"difference: {test_diff}")

# %%
fig, ax = plt.subplots(c - 1, 1)
if c == 2:
    ax.scatter(ds[:, 0], ds[:, -1])
else:
    for i in range(0, c - 1):
        ax[i].scatter(ds[:, i], ds[:, -1])

fig.tight_layout()
plt.show()

# %%
skip = 12
plt.scatter(range(len(ret[skip:])), ret[skip:])
plt.show()
