import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# %%
tf.keras.backend.clear_session()

# %% load data
(train_images_raw, train_labels), (test_images_raw, test_labels) = mnist.load_data()

print(f"train_images.shape: {train_images_raw.shape}")
print(f"train_labels.shape: {train_labels.shape}")
print(f"test_images.shape: {test_images_raw.shape}")
print(f"test_labels.shape: {test_labels.shape}")

# %% display one image
print(train_labels[0])
plt.imshow(train_images_raw[0], cmap="Greys")
plt.show()

# %% reshape data
train_images = train_images_raw.copy().reshape((60000, 28 * 28))
test_images = test_images_raw.copy().reshape((10000, 28 * 28))

print(f"train_images.shape: {train_images.shape}")
print(f"test_images.shape: {test_images.shape}")

# %% map 0-255 to 0-1
train_images = train_images.astype(float) / 255.0
test_images = test_images.astype(float) / 255.0

# %% convert labels to be categorical
train_labels = train_labels.astype(float)
numerical_train_labels = train_labels.copy()
categorical_train_labels = to_categorical(train_labels)

test_labels = test_labels.astype(float)
numerical_test_labels = test_labels.copy()
categorical_test_labels = to_categorical(test_labels)

# %% create model
network = models.Sequential(
    layers=[
        layers.Dense(512, activation="relu", input_shape=(28 * 28,)),
        layers.Dense(10, activation="softmax"),
    ]
)

# noinspection SpellCheckingInspection
network.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics="accuracy",
)

network.summary()

# %% train network
_ = network.fit(train_images, categorical_train_labels, epochs=5, batch_size=128)

# %% test network
test_loss, test_accuracy = network.evaluate(test_images, categorical_test_labels)
print(f"test_accuracy: {test_accuracy}")

# %% get the predictions for training data
probabilities_train = network.predict(train_images)
predict_train = np.argmax(probabilities_train, axis=1)

# %% calculate correct and false
num_correct = (predict_train - train_labels == 0).sum()
num_false = (predict_train - train_labels != 0).sum()
print(f"num_correct: \t{num_correct}")
print(f"num_false:   \t{num_false}")

# %% find the probability difference between the biggest and correct indices
max_probabilities_train = np.max(probabilities_train, axis=1)
# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
correct_probabilities_train = np.take_along_axis(
    probabilities_train, np.expand_dims(train_labels.astype(int), axis=-1), axis=-1
).flatten()
differences_train = (max_probabilities_train - correct_probabilities_train).argsort()

# %% plot the 5 images with the biggest differences
fig1, ax1 = plt.subplots(2, 3)
for i, ax in enumerate(ax1.flatten()):
    image_index = differences_train[-(i + 1)]
    ax.imshow(train_images_raw[image_index], cmap="gray")
    ax.set_title(
        f"expected {train_labels[image_index]}\tpredicted {predict_train[image_index]}"
    )

plt.show()
