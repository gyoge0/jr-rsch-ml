import numpy as np
import matplotlib.pyplot as plt
import csv


# %%


def cost(X, W, Y):
    return ((X.dot(W) - Y) ** 2).mean()


def gradient(X, W, Y):
    err = X.dot(W) - Y
    gradient = X.T.dot(err) / len(Y)
    return gradient


# %%
def read_csv(file):
    lines = []
    with open(file, "r") as f:
        for row in csv.reader(f, delimiter=",", quotechar='"'):
            lines.append(row)
    return lines


m = read_csv("src/intro/7_murder_unemployment.csv")
m = np.asarray(m)[1:, 1:].astype(np.float64)

# %%
m[:, 1:] = (m[:, 1:] - m[:, 1:].mean(axis=0)) / (m[:, 1:].max(axis=0))
X = m[:, 1:]
y = m[:, -1].reshape((20, 1))
W = np.array([0.5, 0.5, 0.5]).reshape((3, 1))

# %%
lr = 0.1
i = 0
while cost(X, W, y) > 0.00000001 and i < 100000:
    i += 1
    W = W - lr * gradient(X, W, y)
    print(f"{i} {cost(X, W, y)}")

print(f"cost: {cost(X, W, y)}")
print(f"weights: {W}")

# %%
print(cost(X, W, y))
