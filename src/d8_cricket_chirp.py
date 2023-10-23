import numpy as np
from d7_murder_unemployment import (
    fit_regression,
    predict,
    mean_square_error,
    create_xy,
    read_csv,
)

cricket = read_csv("C:/Users/875367/Code/jr-rsch-ml/src/d8_cricket_chirp.csv")
cricket = np.asarray(cricket)[1:].astype(np.float64)

cricket = cricket[:, ::-1]

# %%

x, y = create_xy(cricket)
# %%
w = np.ones((2, 1))
print(f"cost: {mean_square_error(x, w, y)}")
w = fit_regression(
    x=x,
    y=y,
    weights=w,
    print_info=False,
    check_interval=1,
)
print(f"cost: {mean_square_error(x, w, y)}")
print(f"weights: {w}")


# %%
best = min(
    dict(
        map(
            lambda lr: (
                lr / 10000,
                fit_regression(
                    x=x,
                    y=y,
                    weights=w,
                    print_info=False,
                    check_interval=1,
                    learning_rate=lr / 100,
                    return_i=True,
                )[1],
            ),
            range(10, 15000, 10),
        )
    ).items(),
    key=lambda lr: lr[0],
)

print(best)
# %%
