# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from tensorflow.keras.datasets import mnist

# %%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
image_index = 4444  # You may select anything up to 60,000
print(train_labels[image_index])

# %% create subplots 4 rows and 2 columns to display 4 different filters
f, ax = plt.subplots(4, 2)
plt.tight_layout()  # This just makes the titles for the plots to space properly

# %% Display the first two images before filters
ax[0, 0].imshow(train_images[image_index], cmap="Greys")
ax[0, 1].imshow(train_images[image_index], cmap="Greys")

# %% vertical filter bright to dark
vertical_filter_bright_to_dark = np.array(
    [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ]
)

res = signal.convolve2d(train_images[image_index], vertical_filter_bright_to_dark)
ax[1, 0].set_title("Vertical B->D")
ax[1, 0].imshow(res, cmap="Greys")

# %% vertical filter dark to bright
vertical_filter_dark_to_bright = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]
)

res = signal.convolve2d(train_images[image_index], vertical_filter_dark_to_bright)
ax[1, 1].set_title("Vertical D->B")
ax[1, 1].imshow(res, cmap="Greys")


# %% horizontal filter bright to dark
horizontal_filter_bright_to_dark = np.array(
    [
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ]
)

res = signal.convolve2d(train_images[image_index], horizontal_filter_bright_to_dark)
ax[2, 0].set_title("Horizontal B->D")
ax[2, 0].imshow(res, cmap="Greys")


# %% horizontal filter dark to bright
horizontal_filter_dark_to_bright = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ]
)

res = signal.convolve2d(train_images[image_index], horizontal_filter_dark_to_bright)
ax[2, 1].set_title("Horizontal D->B")
ax[2, 1].imshow(res, cmap="Greys")


# %% vertical filter sorbel
vertical_filter_sorbel = np.array(
    [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]
)

res = signal.convolve2d(train_images[image_index], vertical_filter_sorbel)
ax[3, 0].set_title("Vertical sorbel")
ax[3, 0].imshow(res, cmap="Greys")


# %% vertical filter scharr
vertical_filter_sorbel = np.array(
    [
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]
)

res = signal.convolve2d(train_images[image_index], vertical_filter_sorbel)
ax[3, 1].set_title("Vertical scharr")
ax[3, 1].imshow(res, cmap="Greys")

# %%
plt.show()
