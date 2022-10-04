# Stephen Huan
import sys
import numpy as np
import cv2 as cv

rng = np.random.default_rng(1)

def binary(img: np.array, points: np.array, rect: bool=False) -> np.array:
    """ Applies Otsu's binarization to a chessboard defined by points.
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html """
    # find bounding box 
    x, y = points[:, 0], points[:, 1]
    i, j, k, l = np.min(x), np.max(x), np.min(y), np.max(y)
    box = img[i:j + 1, k:l + 1]
    # apply blurring then thresholding
    blur = cv.GaussianBlur(box, (5, 5), 0)
    ret, dst = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # overwrite image with bounding box or with convex hull
    if rect:
        img[i:j + 1, k:l + 1] = dst
    else:
        hull = cv.convexHull(points)
        for x in range(box.shape[0]):
            for y in range(box.shape[1]):
                px, py = x + i, y + k
                # point inside the chessboard
                if cv.pointPolygonTest(hull, (px, py), measureDist=False) >= 0:
                    img[px][py] = dst[x][y]
    return img

def binary_iid(img: np.array, points: np.array, rect: bool=False) -> np.array:
    """ Bounding box contains extraneous pixels, sample pixels outside
    of chessboard i.i.d. from chessboard distribution which maintains
    original distribution therefore not affecting Otsu's binarization. """
    # find bounding box 
    x, y = points[:, 0], points[:, 1]
    i, j, k, l = np.min(x), np.max(x), np.min(y), np.max(y)
    box = np.array(img[i:j + 1, k:l + 1])
    # find chessboard mask 
    hull = cv.convexHull(points)
    mask = [[cv.pointPolygonTest(hull, (x + i, y + k), measureDist=False) >= 0
             for y in range(box.shape[1])] for x in range(box.shape[0])]
    mask, on = np.array(mask, dtype=bool), np.sum(mask)
    # generate chessboard pixel distribution
    p = np.zeros(1 << 8, dtype=np.float64)
    for v in box[mask]:
        p[v] += 1
    p /= on
    # replace pixels outside of chessboard with i.i.d. sample
    box[~mask] = rng.choice(256, np.prod(box.shape) - on, p=p)
    # apply blurring then thresholding
    blur = cv.GaussianBlur(box, (5, 5), 0)
    ret, dst = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    img[i:j + 1, k:l + 1][mask] = dst[mask]
    return img

if __name__ == "__main__":
    path = sys.argv[1]
    fname = ".".join(path.split("/")[-1].split(".")[:-1])

    img = cv.imread(path)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    points = np.array([(275, 575), (400, 1025), (400, 200), (725, 725)])
    # points = np.array([(150, 200), (850, 20), (100, 950), (800, 1075)])
    # points = np.array([(0, 0), (0, 1079), (1072, 0), (1072, 1079)])
    # points = np.array([(125, 275), (125, 800), (650, 1050), (650, 20)])
    bimg = binary_iid(grey, points)

    cv.imwrite(f"output/{fname}_iid.png", bimg)

    # cv.imshow("img", bimg)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

