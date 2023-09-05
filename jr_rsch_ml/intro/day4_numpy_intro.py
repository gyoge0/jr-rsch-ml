import numpy as np
import numpy.random as random

# 1,2 create a 20x2 array with random values
rows, cols = (20, 2)
arr = []
for i in range(rows):
    col = []
    for j in range(cols):
        col.append(random.rand())
    arr.append(col)

print(arr)

X = random.rand(20, 2)

# 3 multiply the first column by 20 - multiply by a scalar
X[:, 0] *= 20


# 3 multiply the 2nd column by 1000
X[:, 1] *= 1000
Y = X[:, 0]

# 4 calculate the minimum of column 0
col_0_min = np.min(X[:, 0])

# 4 calculate the max of column 1
col_1_max = np.max(X[:, 1])

# 5 print the max and min:    "min of col 0: xyz,  max of col 1: abc"
print(f"min of col 0: {col_0_min},  max of col 1: {col_1_max}")

# 6 calculate the average of the 1st column
col_1_avg = np.mean(X[:, 0])

# 6 calculate the average of both columns => array of 2 elements
avg_both = np.mean(X, axis=0)

# 7 determine the number of rows and columns in the matrix X
num_rows, num_cols = X.shape

# 8 create a (rows x 1) np array of all zeros using np.zeros -- make sure you specify a tuple
zeroes = np.zeros((num_rows, 1))

# 9 add that np array of all zeros as a third column to X using np.hstack() -- make sure you specify a tuple
X = np.hstack((X, zeroes))

# 10 add column 0 and 1 of X into column 2
X[:, 2] = X[:, 0] + X[:, 1]

# 11 slicing: store a section of rows or columns into a numpy darray
# store rows 3, 4, and 5 into sliceRowsX
slice_rows_x = X[3:6, :]

# 12 store columns 0 and 2 into sliceColsX
slice_rows_y = X[:, (0, 2)]
