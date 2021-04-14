import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from k_means import k_means

def otsu_cv(img: np.array) -> np.array:
    """ Applies Otsu's binarization using openCV's implementation.
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html """
    return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

def histogram(img: np.array, norm: bool=False) -> np.array:
    """ Computes the distribution of the image. """
    p = np.zeros(256, dtype=np.int)
    for x in img.flatten():
        p[x] += 1
    return p if not norm else p/np.sum(p), np.sum(p)

def __threshold(img: np.array, threshold: int) -> np.array:
    """ Binarizes an image given a threshold. """
    return 255*(img > threshold)

def __otsu(img: np.array) -> np.array:
    """ Applies Otsu's binarization to the image with intra-class variance. """
    h, n = histogram(img)
    X0, X1, p = 0, img.sum(), 0
    threshold, best = -1, X1*X1/n
    for t in range(256):
        u1, u0 = t*h[t], h[t]
        X0, X1, p = X0 + u1, X1 - u1, p + u0
        if p > 0 and n - p > 0:
            # divide by -n then add E[X^2] to get the true intra-class variance
            var = X0*X0/p + X1*X1/(n - p)
            if var > best:
                threshold, best = t, var
    return __threshold(img, threshold)

def otsu(img: np.array) -> np.array:
    """ Applies Otsu's binarization to the image with inter-class variance. """
    h, n = histogram(img)
    X0, X1, p = 0, img.sum(), 0
    threshold, best = -1, 0
    for t in range(256):
        u1, u0 = t*h[t], h[t]
        X0, X1, p = X0 + u1, X1 - u1, p + u0
        if p > 0 and n - p > 0:
            # divide by n^2 to get the true inter-class variance
            var = p*(n - p)*(X0/p - X1/(n - p))**2
            if var > best:
                threshold, best = t, var
    return __threshold(img, threshold)

def otsu_vars(img: np.array) -> np.array:
    """ Applies Otsu's binarization to the image to get each variance. """
    h, n = histogram(img)
    X0, X1, p = 0, img.sum(), 0
    l = []
    for t in range(256):
        u1, u0 = t*h[t], h[t]
        X0, X1, p = X0 + u1, X1 - u1, p + u0
        if p > 0 and n - p > 0:
            l.append(p*(n - p)*((X0/p - X1/(n - p))/n)**2)
        else:
            l.append(0)
    return np.array(l)

def otsu_kmeans(img: np.array) -> np.array:
    """ Applies suboptimal Otsu's binarization with k-means. """
    centers, _, _ = k_means(2, [(x,) for x in img.astype(np.float).flatten()])
    threshold = round(np.array(centers).mean())
    return __threshold(img, threshold)

if __name__ == "__main__":
    path = sys.argv[1]
    fname = ".".join(path.split("/")[-1].split(".")[:-1])

    img = cv.imread(path)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # grey = cv.GaussianBlur(grey, (5, 5), 0)

    # histogram visualization 
    p, var = histogram(grey, norm=True)[0], otsu_vars(grey)
    bins = 256
    var /= bins*var.max()/(256*p.max())
    plt.plot(range(256), var, label="inter-class variance")
    t = max(range(256), key=lambda t: var[t])
    plt.plot(t, var[t], 'og')
    plt.hist(range(256), bins=bins, weights=p, label="distribution")
    plt.legend()
    plt.title("Otsu's Binarization Over Image Distribution")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.savefig(f"output/{fname}_hist.png")
    plt.show()

    cv.imshow("", otsu_cv(grey))
    cv.waitKey(0)

    # output
    cv.imwrite(f"output/{fname}_cv.png", otsu_cv(grey))         # 118
    cv.imwrite(f"output/{fname}.png", otsu(grey))               # 118
    cv.imwrite(f"output/{fname}_kmeans.png", otsu_kmeans(grey)) # 119

