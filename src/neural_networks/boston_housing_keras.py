import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import boston_housing

# %%
tf.keras.backend.clear_session()

# %% load data
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# %% standardize data
sc = StandardScaler()
train_data = sc.fit_transform(train_data)
test_data = sc.transform(test_data)

# %% create model
model = models.Sequential(
    layers=[
        layers.Dense(128, activation="relu", input_shape=(13,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ]
)
# noinspection SpellCheckingInspection
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
model.summary()

# %% fit model
history = model.fit(
    x=train_data,
    y=train_labels,
    batch_size=len(train_data),
    epochs=100,
)

# %% plot mae
mae_history = history.history["mae"]
plt.plot(mae_history)
plt.ylabel("mae")
plt.xlabel("epochs")
plt.title("mae vs epochs")
plt.show()

# %% predict test data
test_mse, test_mae = model.evaluate(test_data, test_labels)
predicted_data = model.predict(test_data)


# %%
def manual_mae(predicted, actual):
    return np.mean(np.abs(predicted - actual), axis=0)[0]


test_mae_manual = manual_mae(predicted_data, test_labels.reshape(-1, 1))

# %%
print("Test mae:", test_mae)
print("Test mae manual:", test_mae_manual)
print("Difference:", test_mae - test_mae_manual)
