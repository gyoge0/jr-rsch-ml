import csv
import numpy as np


def mean_square_error(x, w, y):
    return ((x.dot(w) - y) ** 2).mean()


def gradient(x, w, y):
    """
    Calculate the gradient of mean squared error
    :param x: input
    :param w: weights
    :param y: output
    :return: the gradient
    """
    err = x.dot(w) - y
    return x.T.dot(err) / len(y)


def normalize_x(n_x, original):
    """
    Normalizes input data
    :param n_x: input data
    :param original: original dataset
    :return: normalized data
    """
    o_std = np.std(original[:, :-1], axis=0)
    o_mean = np.mean(original[:, :-1], axis=0)
    n_normal = (n_x - o_mean) / o_std
    return n_normal, o_mean, o_std


def create_xy(original):
    """
    Creates input and output arrays
    :param original: original data
    :return: input and output data
    """
    x = original[:, :-1]
    x, _, _ = normalize_x(x, original)
    bias = np.ones((len(x), 1))
    x = np.hstack((bias, x))
    y = original[:, -1][:, None]
    return x, y


def fit_regression(
    x,
    y,
    weights,
    max_iterations=1000000,
    learning_rate=0.1,
    goal_error=0.001,
    precision=7,
    print_info=False,
    check_interval=1,
):
    """
    Fits a linear regression
    :param x: input data
    :param y: output data
    :param weights: initial weights
    :param max_iterations: maximum iterations
    :param learning_rate: learning rate
    :param goal_error: goal error
    :param precision: decimals to check when determining if error has stopped changing
    :param print_info: if info should be printed while fitting
    :param check_interval: how many iterations between checking to stop
    :return: the final weights
    """
    last_mse = -1
    i = 0
    while mean_square_error(x, weights, y) > goal_error and i < max_iterations:
        i += 1
        weights = weights - learning_rate * gradient(x, weights, y)
        if i % check_interval == 0:
            if print_info:
                print(f"{i} {mean_square_error(x, weights, y)}")

            if round(mean_square_error(x, weights, y), precision) == last_mse:
                break
            else:
                last_mse = round(mean_square_error(x, weights, y), precision)

    if not print_info:
        pass
    elif i >= 100000:
        print("reached max iterations")
    elif mean_square_error(x, weights, y) <= 0.001:
        print("reached goal error")
    else:
        print("weights not changing")

    return weights


def predict(input_data, weights, original):
    """
    Predict using a linear regression
    :param input_data: unnormalized input data
    :param weights: weights from regression
    :param original: original data
    :return:
    """
    input_data, _, _ = normalize_x(input_data, original)
    bias = np.ones((len(input_data), 1))
    input_data = np.hstack((bias, input_data))
    pred = input_data.dot(weights)
    _, y = create_xy(original)
    error = mean_square_error(input_data, weights, y)
    return pred, error


def read_csv(file):
    lines = []
    with open(file, "r") as f:
        for row in csv.reader(f, delimiter=",", quotechar='"'):
            lines.append(row)
    return lines


def _main():
    # murder_unemployment = read_csv("src/d7_murder_unemployment.csv")
    murder_unemployment = read_csv(
        "C:/users/875367/Code/jr-rsch-ml/src/d7_murder_unemployment.csv"
    )
    murder_unemployment = np.asarray(murder_unemployment)[1:, 2:].astype(np.float64)

    # %%

    x, y = create_xy(murder_unemployment)
    # %%
    w = np.array([2, 1, 1]).reshape((3, 1))
    print(f"cost: {mean_square_error(x, w, y)}")
    w = fit_regression(
        x=x,
        y=y,
        weights=np.array([2, 1, 1]).reshape((3, 1)),
        learning_rate=0.94,
        print_info=True,
    )

    # %%
    print(f"cost: {mean_square_error(x, w, y)}")
    print(f"weights: {w}")

    # %%

    prediction = predict(np.array([22.4, 8.6]).reshape((1, 2)), w, murder_unemployment)
    print(prediction)


if __name__ == "__main__":
    _main()
