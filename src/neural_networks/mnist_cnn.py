import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# %%
# noinspection DuplicatedCode
tf.keras.backend.clear_session()

# %% load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"train_images.shape: {train_images.shape}")
print(f"train_labels.shape: {train_labels.shape}")
print(f"test_images.shape: {test_images.shape}")
print(f"test_labels.shape: {test_labels.shape}")

# %% display one image
print(train_labels[0])
plt.imshow(train_images[0], cmap="Greys")
plt.show()

# %% map 0-255 to 0-1
# noinspection DuplicatedCode
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
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# noinspection SpellCheckingInspection,DuplicatedCode
network.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics="accuracy",
)

network.summary()

# %% train network
history = network.fit(train_images, categorical_train_labels, epochs=5, batch_size=128)

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
# max_probabilities_train = np.max(probabilities_train, axis=1)
# # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
# correct_probabilities_train = np.take_along_axis(
#     probabilities_train, np.expand_dims(train_labels.astype(int), axis=-1), axis=-1
# ).flatten()
# differences_train = (max_probabilities_train - correct_probabilities_train).argsort()


# %% find the images we got wrong and we were confident about
def select_worst_best(actual_values, predicted_values, predicted_probabilities, images):
    # Create a numpy array of max probabilities (using prob)
    max_prob = np.max(predicted_probabilities, axis=1)

    # Stack the predictions, actuals and probabilities with images
    stacked = np.hstack(
        (
            actual_values.reshape(-1, 1),
            predicted_values.reshape(-1, 1),
            max_prob.reshape(-1, 1),
            images,
        )
    )

    stacked_correct = stacked[stacked[:, 0] == stacked[:, 1]]
    stacked_wrong = stacked[stacked[:, 0] != stacked[:, 1]]
    stacked_correct = stacked_correct[stacked_correct[:, 2].argsort()]
    stacked_wrong = stacked_wrong[stacked_wrong[:, 2].argsort()]

    most_correct_guesses = stacked_correct[-9:]
    most_incorrect_guesses = stacked_wrong[-9:]
    return most_incorrect_guesses, most_correct_guesses


worst_guesses, best_guesses = select_worst_best(
    numerical_train_labels,
    predict_train,
    probabilities_train,
    train_images,
)


# %% plot the 9 worst images
fig_worst, ax_worst = plt.subplots(3, 3, constrained_layout=True)
for i, aw in enumerate(ax_worst.flatten()):
    aw.imshow(worst_guesses[i, 3:].reshape((28, 28)), cmap="gray")
    aw.set_title(f"expected {worst_guesses[i, 0]} predicted {worst_guesses[i, 1]}")
fig_worst.suptitle("Worst")
fig_worst.set_size_inches(8, 6)
plt.savefig("figs/worst_mnist.png")
plt.show()

# %% plot the 9 best images
fig_best, ax_best = plt.subplots(3, 3, constrained_layout=True)
for i, ab in enumerate(ax_best.flatten()):
    ab.imshow(best_guesses[i, 3:].reshape((28, 28)), cmap="gray")
    ab.set_title(best_guesses[i, 0])
fig_best.suptitle("Best")
fig_best.set_size_inches(8, 6)
plt.savefig("figs/best_mnist.png")
plt.show()
