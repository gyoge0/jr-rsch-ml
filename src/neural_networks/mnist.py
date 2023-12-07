import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

# %% load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"train_images.shape: {train_images.shape}")
print(f"train_labels.shape: {train_labels.shape}")
print(f"test_images.shape: {test_images.shape}")
print(f"test_labels.shape: {test_labels.shape}")

# %% display one image
image_index = 0
print(train_labels[image_index])
plt.imshow(train_images[image_index], cmap="Greys")
plt.show()

# %% reshape data
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

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
