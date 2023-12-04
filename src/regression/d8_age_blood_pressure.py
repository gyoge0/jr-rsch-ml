import numpy as np

from linear_regression import (
    fit_regression,
    mean_square_error,
    create_xy,
    read_csv,
)

blood_pressure = read_csv(
    "C:/Users/875367/Code/jr-rsch-ml/src/d8_age_blood_pressure.csv"
)
blood_pressure = np.asarray(blood_pressure)[1:, 2:].astype(np.float64)


# %%

x, y = create_xy(blood_pressure)
# %%
w = np.ones((2, 1))
print(f"cost: {mean_square_error(x, w, y)}")
w = fit_regression(
    x=x,
    y=y,
    weights=w,
    print_info=True,
    check_interval=1,
    learning_rate=0.01,
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
                    learning_rate=lr / 1000,
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
