import numpy as np
from matplotlib import pyplot as plt


# noinspection DuplicatedCode
def create_actuals():
    # function to create an array of x values
    # and y values with a given slope and y intercept
    # data = np.array((numPoints, 2)).astype(float)
    # no need to change
    data = np.zeros((num_points, 3))

    for k in range(num_points):
        data[k][0] = 1.0
        data[k][1] = float(k)
        data[k][2] = y_intercept + float(slope) * k

    x = data[:, 0:2]
    y = data[:, 2]
    print(x, y)
    return x, y


def create_weights():
    # this function creates a set of weights that you
    # will calculate the cost for
    # no need to change
    num_weights = 30
    bias_weights = np.linspace(0, y_intercept * 2, num_weights)
    x1_weights = np.linspace(0, slope * 2, num_weights)
    weights = np.transpose([bias_weights, x1_weights])
    return weights


def calc_cost(x, weights, y):
    # =============================================================================
    # ############  YOUR CODE HERE
    # return the cost for the weights
    # =============================================================================
    return np.sum((x.dot(weights) - y) ** 2) / len(x)


def scatter_weights(weights, cost_arr):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(weights[:, 0], weights[:, 1], cost_arr)
    ax.set_xlabel("m")
    ax.set_ylabel("b")
    ax.set_zlabel("cost")
    plt.show()


# Main
num_points = 4
slope = 8.0
y_intercept = 10.0


def _main():
    x, y = create_actuals()
    weight_arr = create_weights()
    cost_arr = np.array(list(map(lambda w: calc_cost(x, w, y), weight_arr)))
    print(cost_arr)
    print(f"Best weights: {weight_arr[cost_arr.argmin(), :]}")
    scatter_weights(weight_arr, cost_arr)


if __name__ == "__main__":
    _main()
