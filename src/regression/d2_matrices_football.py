# Importing the libraries
import numpy as np


# Importing the datasets\linear

# =============================================================================
# ==> RETURN THE NAME OF THE QB WITH THE MAX NUMBER OF ATTEMPTS AND THE ATTEMPT
#     COUNT
# =============================================================================


# =============================================================================
# Read in the matrix1999 file
# =============================================================================


def max_num_of_attempts():
    matrix1998 = np.loadtxt("day4_matrices_football_1998.csv", delimiter=",", dtype=str)
    matrix1999 = np.loadtxt("day4_matrices_football_1999.csv", delimiter=",", dtype=str)
    matrix1998_no_names = matrix1998[1:, :]
    matrix1999_no_names = matrix1999[1:, :]
    # =============================================================================
    # Strip off the names into a separate names list
    # =============================================================================
    matrix1998_names = matrix1998_no_names[:, 0]
    matrix1999_names = matrix1999_no_names[:, 0]
    # =============================================================================
    # note that matrix1998 and matrix 1999 is a mixed type and the
    # numbers must be converted to floats - convert them to floats
    # =============================================================================
    matrix1998_no_names = matrix1998_no_names[:, 1:].astype(float)
    matrix1999_no_names = matrix1999_no_names[:, 1:].astype(float)
    # =============================================================================
    # Find the Difference and print
    # =============================================================================
    dif = matrix1999_no_names - matrix1998_no_names
    print(dif)
    # =============================================================================
    # print 2 year total
    # =============================================================================
    _sum = matrix1999_no_names + matrix1998_no_names
    print(_sum)
    # =============================================================================
    # print average of two years
    # =============================================================================
    avg = (matrix1999_no_names + matrix1998_no_names) / 2
    print(avg)
    # =============================================================================
    # Who had and what was the amount of the max attempts for 1998
    # max values of each column (axis = 0) and print it out
    # ==> RETURN THE NAME OF THE QB WITH THE MAX NUMBER OF ATTEMPTS AND THE ATTEMPT
    #     COUNT
    # =============================================================================
    name, attempts = matrix1998[matrix1998_no_names[:, 0].argmax() + 1, :2]
    print(matrix1998_no_names.max(axis=0))
    # MaxValueByColum
    # MaxIndexByColumn
    return name, str(attempts)


def dot_product():
    # =============================================================================
    #  Solve the dot product of the following
    #  From the Matrix packet Flower Arrangements
    #  Carl wants to buy flowers for two of his friends. Inga prefers mostly roses and
    #  Tasha prefers mostly lilies. Carl wants to buy two different arrangements.
    #  Inga’s arrangement will include six roses, four carnations, and two lilies.
    #  Tasha’s arrangement will include three roses, three carnations, and six lilies.
    #  As in question
    #  #1, Rose’s Flower Shop charges $2.25 for a rose, $1.25 for a carnation,
    #  and $1.95 for a lily. How much will each arrangement cost?
    #
    #  ==> RETURN THE ARRAY RESULTING FROM THE DOT PRODUCT
    # =============================================================================
    # AB = X
    amounts = np.array(
        [
            [6, 4, 2],
            [3, 3, 6],
        ],
    )
    prices = np.array([2.25, 1.25, 1.95])
    return amounts.dot(prices)


# noinspection PyPep8Naming
def solve_equation():
    # solve equation with linalg
    # Complete first example (soda, hot dogs and candy bar prices)
    # =============================================================================
    # System of equations:
    #  x + 2y +2z = 4.35 (Albert)
    # 2x + 3y + z = 4.40 (Megan)
    #  x +  3z = 4.50 (you)
    #
    # ==> RETURN THE SOLUTION MATRIX
    # =============================================================================
    # AX = B
    A = np.array(
        [
            [1, 2, 2],
            [2, 3, 1],
            [1, 0, 3],
        ]
    )
    X = np.array([4.35, 4.40, 4.50])
    return np.linalg.solve(A, X)


# =============================================================================
# np.linalg.solve(A, X) Methods go above
# =============================================================================
qB, max_attempts = max_num_of_attempts()
print(qB, "had the maximum number of attempts in 1998 of", max_attempts)
print("dot product array: ", dot_product())
print("solution to equations: ", solve_equation())
