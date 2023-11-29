import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_error(x, w, y):
    sig = sigmoid(x.dot(w))
    cost = -1 * (y * np.log(sig) + (1 - y) * np.log(1 - sig))
    return np.mean(cost)


def gradient(x, w, y):
    err = sigmoid(x.dot(w)) - y
    h, _ = w.shape
    return (err.T.dot(x) / len(y)).T


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


def argmax_predict(x, w):
    return np.argmax(x.dot(w), axis=1)


def calc_confusion_matrix(p, y, print_info=False):
    tpc = (p[y == 1] == 1).sum()
    tnc = (p[y == 0] == 0).sum()
    fpc = (p[y == 0] == 1).sum()
    fnc = (p[y == 1] == 0).sum()

    total_pos = (y == 1).sum()
    total_neg = (y == 0).sum()
    accuracy = (tpc + tnc) / (total_pos + total_neg)
    recall = tpc / (tpc + fnc)
    precision = tpc / (tpc + fpc)

    f_score = (2 * precision * recall) / (precision + recall)

    info = {
        "TPC": tpc,
        "TNC": tnc,
        "FPC": fpc,
        "FNC": fnc,
        "TP": total_pos,
        "TN": total_neg,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
    }

    if print_info:
        print(
            "\nTrue Positives: \t\t",
            tpc,
            "\nPercent Positives correct:\t",
            f"{(tpc/total_pos):.0%}\n",
        )
        print(
            "False Positives: \t\t",
            fpc,
            "\nPercent Positives Incorrect:\t",
            f"{(fpc/total_pos):.0%}\n",
        )
        print(
            "True Negatives: \t\t",
            tnc,
            "\nPercent Negatives correct:\t",
            f"{(tnc/total_neg):.0%}\n",
        )
        print(
            "False Negatives: \t\t",
            fnc,
            "\nPercent Negatives incorrect:\t",
            f"{(fnc/total_neg):.0%}\n",
        )

        print("\nPrecision: \t\t\t", f"{precision:.0%}")
        print("Recall:\t\t\t\t", f"{recall:.0%}")
        print("\nF1: \t\t\t\t", f"{f_score:.0%}")

        print("Accuracy: \t\t\t", f"{accuracy:.0%}")

    return info


def initial_weights(x, num=1):
    _, h = x.shape
    w = np.repeat(0, h * num).reshape((h, num))
    return w
