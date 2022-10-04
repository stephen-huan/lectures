"""
downscsaling: https://github.com/posva/catimg/blob/master/src/sh_image.c
averages windows instead of ffmpeg
"""
import os, subprocess
from PIL import Image
import colorlib

TEMP = "temp.png"

def change_size(path: str, height: int, width: int, alg: str) -> Image:
    """ Changes the image to height x width. """
    # use ffmpeg to scale image, see https://trac.ffmpeg.org/wiki/Scaling
    subprocess.call(["ffmpeg", "-i", path, "-vf", f"scale={width}:{height}",
                     "-sws_flags", alg, TEMP, "-loglevel", "quiet"])
    im = Image.open(TEMP).convert("RGB")
    os.remove(TEMP)
    return im

def get_size(size: tuple, h: int, w: int) -> tuple:
    """ Adjusts the size, taking into account aspect ratio. """
    aspect_ratio = size[0]/size[1]
    if h == -1:
        return (round(w/aspect_ratio), w)
    if w == -1:
        return (h, round(h*aspect_ratio))

def centroid(points: list) -> tuple:
    """ Returns the centroid of a list of points. """
    denom, D = len(points), len(points[0])
    return tuple(sum(p[d] for p in points)/denom for d in range(D))

def box(grid: list, i: int, j: int, hf: float, wf: float) -> tuple:
    """ Finds the average value in the box. """
    N, M = len(grid), len(grid[0])
    t1, t2 = lambda x: round(hf*x), lambda x: round(wf*x)
    return centroid([grid[k][l] for l in range(t2(j), min(t2(j + 1), M))
                     for k in range(t1(i), min(t1(i + 1), N))])

def downscale(fname: str, h: int, w: int, space: str="lab") -> list:
    """ Downscales an image to the proper height and width.
    Assumes the source image is larger than the target. """
    im = Image.open(fname).convert("RGB")
    N, M, data = im.size[0], im.size[1], list(im.getdata())
    grid = [[colorlib.convert(data[x + y*N], "srgb", space)
             for x in range(N)] for y in range(M)]

    h, w = get_size(im.size, h, w)
    hf, wf = M/h, N/w
    return [[box(grid, i, j, hf, wf) for j in range(w)] for i in range(h)]

if __name__ == "__main__":
    space = "lab"
    fname = "/Users/stephenhuan/Pictures/anime/k_on/azusa_cropped.png"
    m = downscale(fname, -1, 40, space)

    N, M = 223, 348
    N, M = 40, 62
    img = Image.new("RGB", (N, M))
    img.putdata([colorlib.convert(m[x][y], space, "srgb")
                 for x in range(M) for y in range(N)])
    img.save("test.png")

