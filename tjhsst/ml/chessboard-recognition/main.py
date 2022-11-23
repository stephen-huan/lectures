# Stephen Huan
import sys, os
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from grid import WIDTH, HEIGHT
from binary import binary_iid

def render_points(img: np.array, points: np.array) -> np.array:
    """ Draws a circle at each point on a copy of the image. """
    out = img.copy()
    for i in points:
        x, y = i.ravel()
        cv.circle(out, (x, y), 1, (0, 0, 255), -1)
    return out

if __name__ == "__main__":
    path = sys.argv[1]
    fname = ".".join(path.split("/")[-1].split(".")[:-1])
    out = f"output/{fname}"
    if not os.path.exists(out):
        os.mkdir(f"output/{fname}")

    # load image and greyscale
    img = cv.imread(path)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get initial corners
    corners = cv.goodFeaturesToTrack(grey, round(1.5*81), 0.01, 10)
    corners = np.int0(corners)
    cv.imwrite(f"output/{fname}/1.jpg", render_points(img, corners))

    # render binary image
    binary = np.zeros((WIDTH, HEIGHT), np.float)
    for i in corners:
        x, y = i.ravel()
        binary[y][x] = 1
    cv.imwrite(f"output/{fname}/2.jpg", 255*binary)

    # load pre-trained neural network
    model = keras.models.load_model(f"models/model{WIDTH}x{HEIGHT}")
    yp = model.predict(binary.reshape(1, WIDTH, HEIGHT, 1))
    yp = yp[0].reshape(256, 256)
    cv.imwrite(f"output/{fname}/3.jpg", tf.math.round(255*yp).numpy())

    # take top 81 intensities
    threshold = sorted(yp.ravel(), reverse=True)[81]
    points = np.array([(x, y) for x in range(WIDTH) for y in range(HEIGHT)
                       if yp[y][x] > threshold])
    cv.imwrite(f"output/{fname}/4.jpg", render_points(img, points))

    # binarize chessboard with final points
    cv.imwrite(f"output/{fname}/5.jpg", binary_iid(grey.T, points).T)

