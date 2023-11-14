import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_error(x, w, y):
    sig = sigmoid(x.dot(w))
    cost = -1 * (y * np.log(sig) + (1 - y) * np.log(1 - sig))
    return np.mean(cost)


def gradient(x, w, y):
    err = sigmoid(x.dot(w)) - y
    return err.dot(x) / len(y)


def fit_regression(
    x,
    y,
    weights,
    max_iterations=1000000,
    learning_rate=0.1,
    goal_error=0.000000001,
    precision=7,
    print_info=False,
    check_interval=1,
    return_i=False,
    ret=False,
):
    last_mse = -1
    i = 0
    cost = []
    while logistic_error(x, weights, y) > goal_error and i < max_iterations:
        i += 1
        weights = weights - learning_rate * gradient(x, weights, y)
        cost.append(logistic_error(x, weights, y))
        if i % check_interval == 0:
            if print_info:
                print(f"{i} {logistic_error(x, weights, y)}")

            if round(logistic_error(x, weights, y), precision) == last_mse:
                break
            else:
                last_mse = round(logistic_error(x, weights, y), precision)

    if not print_info:
        pass
    elif i >= 100000:
        print("reached max iterations")
    elif logistic_error(x, weights, y) <= goal_error:
        print("reached goal error")
    else:
        print("weights not changing")

    if ret:
        return weights, cost

    if return_i:
        return weights, i
    else:
        return weights
