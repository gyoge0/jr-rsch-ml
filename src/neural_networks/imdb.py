import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

# %%
tf.keras.backend.clear_session()

# %% override np.load for when keras.datasets.reuters.load_data calls it
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True)

# %% load data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(f"Example: {train_data[0]}")
print(f"Shape: {train_labels.shape}")
print(f"Max: {max([max(sequence) for sequence in train_data])}")

# %% decode review
# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decode the review
# indices were offset by 3
# 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])

print(f"Decoded review: {decoded_review}")


# %% vectorize data
# We cannot feed lists of integers into a neural network. We have to turn our
# lists into tensors.
#
# We could one-hot-encode our lists to turn them into vectors of 0s and 1s.
# Concretely, this would mean, for instance, turning the sequence [3, 5] into a
# 10,000-dimensional vector that would be all-zeros except for indices 3 and 5,
# which would be ones. Then we could use as first layer in our network a Dense
# layer, capable of handling floating point vector data.
# We will go with the latter solution. Let's vectorize our data, which we will
# do manually for maximum clarity:
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0  # set specific indices of results[i] to 1s
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype(float)
y_test = np.asarray(test_labels).astype(float)

print("x_train[0]:", x_train[0])

# %% create validation data
x_val = x_train[:10_000]
x_train = x_train[10_000:]
y_val = y_train[:10_000]
y_train = y_train[10_000:]

# %% create model
model = models.Sequential(
    layers=[
        layers.Dense(128, activation="relu", input_shape=(10_000,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2, input_shape=(32,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# noinspection SpellCheckingInspection
model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# %% train network
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=512,
    epochs=3,
    validation_data=(x_val, y_val),
)

# %%
results = model.evaluate(x_train, y_train)
print(f"train: {results}")
results = model.evaluate(x_val, y_val)
print(f"val: {results}")
results = model.evaluate(x_test, y_test)
print(f"test: {results}")

# %%
history_dict = history.history
loss = history_dict["loss"]
val_loss = history_dict["loss"]
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
epochs = range(1, len(acc) + 1)

fig1, ax1 = plt.subplots()
ax1.plot(epochs, loss, "bo", label="Training loss")
# b is for "solid blue line"
ax1.plot(epochs, val_loss, "b", label="Validation loss")
ax1.set(xlabel="Epochs", ylabel="Loss", title="Training and validation loss")
ax1.legend()
acc_values = history_dict["accuracy"]
val_acc_values = history_dict["val_accuracy"]

plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(epochs, acc, "ro", label="Training acc")
ax2.plot(epochs, val_acc, "b", label="Validation acc")
ax2.set(xlabel="Epochs", ylabel="Loss", title="Training and validation accuracy")
ax2.legend()

plt.show()

# %%

prediction = model.predict(x_test)
