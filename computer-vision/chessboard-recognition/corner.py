# Stephen Huan
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from grid import train_pair, WIDTH, HEIGHT

np.random.seed(1)
tf.random.set_seed(1)

# generate dataset
N, f, n = 10**4, 1/4, 50 # number of pairs, max removal, max number of points
X, y = map(np.array, zip(*[train_pair(f, n) for i in range(N)]))
# scale to between 0 and 1, add channel
X, y = X.reshape(*X.shape, 1)/255, y.reshape(*y.shape, 1)/255
# flatten y into a vector
# y = y.reshape(y.shape[0], np.prod(y.shape[1:]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5)

# create model
kwargs = {"padding": "same", "activation": "relu"}
model = keras.models.Sequential([
    # downscale
    layers.Conv2D(16, (3, 3), input_shape=X_train[0].shape, **kwargs),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), **kwargs),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), **kwargs),
    # upscale
    layers.Conv2DTranspose(16, (3, 3), strides=2, **kwargs),
    layers.Conv2D(8, (3, 3), **kwargs),
    layers.Conv2DTranspose(4,  (3, 3), strides=2, **kwargs),
    # flatten to output with 1D convolution, make sure between 0 and 1
    # sigmoid doesn't work too well, use tanh(relu(x))
    layers.Conv2D(1, (1, 1), activation="tanh"),
    layers.ReLU(),
    # layers.Flatten(),
])
print(model.summary())

# compile and train model
model.compile(optimizer="adam",
              # loss="binary_crossentropy",
              loss="mean_squared_error",
              metrics=["binary_accuracy",
                       keras.metrics.Precision(),
                       keras.metrics.Recall()
              ]
             )
model.fit(X_train, y_train, epochs=2)
print(model.evaluate(X_test, y_test))

# save model
model.save(f"models/model{WIDTH}x{HEIGHT}")

# write sample images
yp = tf.math.round(255*model.predict(X_test))
for i in range(10):
    k = str(i).rjust(1, "0")
    cv.imwrite(f"output/nn/{k}_gridX.png", 255*X_test[i])
    cv.imwrite(f"output/nn/{k}_gridy.png", 255*y_test[i])
    cv.imwrite(f"output/nn/{k}_gridP.png", yp[i].numpy())

