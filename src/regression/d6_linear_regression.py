import numpy as np
from matplotlib import pyplot as plt


# =============================================================================
# Complete the code for linear regression
# 1a, 1b, 2, 3, 4
# Code should be vectorized when complete
# =============================================================================


def createData():
    data = np.array(
        [
            [1, 0, 1],
            [1, 1, 1.5],
            [1, 2, 2],
            [1, 3, 2.5],
        ]
    ).astype(float)
    # 1a
    x = data[:-1, :]
    y = data[-1, :]
    return x, y


# 2
# noinspection PyPep8Naming,PyShadowingNames
def calcCost(X, W, Y):
    return ((X.dot(W) - Y) ** 2).mean()


# 4
# noinspection PyPep8Naming,PyShadowingNames
def calcGradient(X, Y, W):
    err = X.dot(W) - Y
    return X.T.dot(err) / len(Y)


############################################################
# Create figure objects
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]

# 1b
# =============================================================================
#  X,Y use createData method to create the X,Y matrices
# Weights - Create initial weight matrix to have the same weights as features
# Weights - should be set to 0
# =============================================================================
#
X, Y = createData()
numRows, numCols = X.shape
W = np.zeros((numCols, 1))

# set learning rate - the list is if we want to try multiple LR's
# We are only doing one of them today
lrList = [0.3, 0.01]
lr = lrList[0]

# set up the cost array for graphing
costArray = [calcCost(X, W, Y)]

# initalize while loop flags
finished = False
count = 0

while not finished and count < 10:
    gradient = calcGradient(X, Y, W)
    # print(gradient)
    # 5 update weights
    W = W - lr * gradient

    # print("weights: ", W)
    costArray.append(calcCost(X, W, Y))
    lengthOfGradientVector = np.linalg.norm(gradient)
    if lengthOfGradientVector < 0.00001:
        finished = True
    count += 1

print("weights: ", W)
ax.plot(np.arange(len(costArray)), costArray, "ro", label="cost")


ax.set_title("Cost as weights are changed")
ax.set_xlabel("iteration")
ax.set_ylabel("Costs")
ax.legend()
plt.show()
