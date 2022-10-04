# Stephen Huan
import numpy as np
import cv2 as cv

rng = np.random.default_rng(1)
# WIDTH, HEIGHT = 1920, 1080 # image height and width
WIDTH, HEIGHT = 256, 256
N = 9                      # width of grid 
pi, cos, sin = np.pi, np.cos, np.sin

def rot_matrix(angle: tuple) -> np.array:
    """ Constructs a rotation matrix for the given angle. """
    tx, ty, tz = angle
    matX = np.array([[1, 0, 0],
                     [0, cos(tx), -sin(tx)],
                     [0, sin(tx),  cos(tx)]])
    matY = np.array([[ cos(ty), 0, sin(ty)],
                     [0, 1, 0],
                     [-sin(ty), 0, cos(ty)]])
    matZ = np.array([[cos(tz), -sin(tz), 0],
                     [sin(tz),  cos(tz), 0],
                     [0, 0, 1]])
    return matX@matY@matZ

def scale(point: tuple) -> tuple:
    """ Scales and translates a point. """
    return (round(point[0] + HEIGHT/2),
            round(point[1] +  WIDTH/2))

def render_grid(dist: float, angle: tuple) -> list:
    """ Renders the grid at the given distance and orientation. """
    # construct and center 9x9 grid of corners
    grid = np.array([(i, j, 0) for i in range(N)
                     for j in range(N)], np.float64).T
    grid -= (np.sum(grid, axis=1)/grid.shape[1]).reshape((3, 1))
    # rendering parameters: https://en.wikipedia.org/wiki/3D_projection
    camera = np.array([0, 0, 10])   # location of the eye
    camera = camera.reshape((3, 1)) # make column vector
    screen = np.array([0, 0, dist]) # display surface relative to camera
    # transformation matrix
    T = np.array([[1, 0, screen[0]/screen[2]],
                  [0, 1, screen[1]/screen[2]],
                  [0, 0,         1/screen[2]]])
    # project 3D object to 2D plane
    projected = T@(rot_matrix(angle)@grid - camera)
    return list(map(scale, ((point[0]/point[2], point[1]/point[2])
                    for point in map(lambda i: projected[:, i], range(N*N)))))

def to_image(points: list) -> np.array:
    """ Converts a list of points into an image. """
    img = np.zeros((HEIGHT, WIDTH), np.uint8)
    for point in points:
        x, y = point
        if 0 <= x < HEIGHT and 0 <= y < WIDTH:
            img[x][y] = 255
    return img

def randrange(a: float, b: float) -> float:
    """ Generates a random number in the range [a, b). """
    return (b - a)*rng.random() + a

def random_grid() -> np.array:
    """ Generate a random grid for training. """
    # sample parameters --- between 50 to 100 pixels, 45 degree rotations
    # dist = randrange(500, 910)
    dist = randrange(HEIGHT*25/64, HEIGHT*25/32)
    angle = tuple(randrange(-pi/4, pi/4) for i in range(3))
    # generate grid and translate such that the grid is in the image
    points = np.array(render_grid(dist, angle))
    x, y = points[:, 0], points[:, 1]
    i, j, k, l = np.min(x), np.max(x), np.min(y), np.max(y)
    return points + np.array([rng.integers(-i, HEIGHT - j),
                              rng.integers(-k, WIDTH - l)])

def train_pair(f: float=1/2, n: int=100) -> tuple:
    """ Return the training pair (X, y). """
    y = random_grid()
    # remove points from the original grid
    X = rng.permutation(y)
    X = X[:round((1 - randrange(0, f))*len(y))]
    # add new random points
    m = rng.integers(n)
    px, py = rng.integers(HEIGHT, size=m), rng.integers(WIDTH, size=m)
    X = np.append(X, np.array([px, py]).T, axis=0)
    return to_image(X), to_image(y)

if __name__ == "__main__":
    X, y = train_pair()
    cv.imwrite(f"output/gridX.png", X)
    cv.imwrite(f"output/gridy.png", y)

