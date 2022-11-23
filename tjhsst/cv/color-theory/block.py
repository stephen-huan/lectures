import sys, argparse, random
import colorlib
from colorlib import esc_color, color_esc, flatten
from k_means import k_means, closest, dist
# merges color codes when possible to speedup processing
# each color two numbers; first number top and second number bottom

SQUARE = "▀"       # character used as a pixel
LAST = "[?25h" # ending token
EOL = "[m"     # stop a color code

### helper functions

def open_image(fname: str) -> list:
    """ Opens a text file into a grid of tuples. """
    with open(fname) as f:
        grid = [line for line in f]
    return [list(map(esc_color, row.split(SQUARE)[:-1])) for row in grid[:-1]]

def color_rgb(c: tuple) -> tuple:
    """ Convert a two length ansi sequence to a six length rgb tuple. """
    return to_color(c[0]) + to_color(c[1])

def rgb_color(c: tuple) -> tuple:
    """ Convert a six length rgb tuple to a two length ansi sequence. """
    return (from_color(c[:3]), from_color(c[3:])) \
        if "ansi" not in args.color else from_color(c)

def from_color(c: tuple) -> int:
    """ Finds the closest ansi color to the given color,
        being careful to only pick possible ansi colors. """
    if "ansi" in args.color:
        return tuple(map(round, c))
    c = colorlib.conv[args.color][args.diff](c)
    return colorlib.rectify(c, args.diff, metric)

if __name__ == "__main__":
    # color space transformation
    spaces, metrics = colorlib.conv["ansi256"], colorlib.metrics

    parser = argparse.ArgumentParser(description="Optimize text from catimg.")
    parser.add_argument("path", metavar="PATH",
                        help="path to ascii image")
    parser.add_argument("-c", "--color", default="cam02ucs", choices=spaces.keys(),
                        help="color space, default cam02ucs")
    parser.add_argument("-d", "--diff", default="cam02ucs", choices=spaces.keys(),
                        help="color space for difference metric, default cam02ucs")
    parser.add_argument("-m", "--metric", default="euclidean", choices=metrics.keys(),
                        help="color difference metric, default euclidean")
    parser.add_argument("-p", "--power", action="store_true",
                        help="use the power transform")
    parser.add_argument("-k", "--num", metavar="K", type=int,
                        help="number of colors")
    parser.add_argument("-f", "--file",
                        help="load color scheme from file")
    parser.add_argument("-r", "--row", action="store_true",
                        help="run k-means on rows instead of the entire image")
    parser.add_argument("-s", "--seed", type=int, default=1,
                        help="set random seed")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output")
    args = parser.parse_args()

    random.seed(args.seed)
    im = open_image(args.path)

    # color space projection
    to_color, m = spaces[args.color], args.metric
    metric = colorlib.power(args.color, m) if args.power else metrics[m]

    if args.num is not None:
        K = args.num  # use different picture for color scheme
        im2 = open_image(args.file) if args.file is not None else im
        if args.row:
            im_new = []
            for row in im:
                rgb = list(map(color_rgb, row))
                centers, ids, groups = k_means(min(K, len(set(rgb))), rgb)
                px = [rgb_color(c) for c in centers]
                im_new.append([px[ids[color_rgb(c)]] for c in row])
            im = im_new
        else:
            # convert image to rgb
            rgb = list(map(color_rgb, flatten(im2)))
            # run k-means
            centers, ids, groups = k_means(min(K, len(set(rgb))), rgb)
            px = [rgb_color(c) for c in centers]
            # process image
            if args.file is not None:
                D = {center: i for i, center in enumerate(centers)}
                im = [[px[D[closest(centers, color_rgb(c))]]
                    for c in row] for row in im]
            else:
                im = [[px[ids[color_rgb(c)]] for c in row] for row in im]

    # merge same color codes
    count = 0
    for row in im:
        new, curr = [], (float("inf"),)*6
        for token in row:
            new.append("" if token == curr else token)
            curr = token
        new = list(map(lambda c: color_esc(c) if c != "" else "", new))
        print(SQUARE.join(new) + SQUARE + EOL, end="")
        print(f" {new.count('')/len(new):.1%}" if args.verbose else "")
        count += new.count("")
    print(LAST, end="")

    if args.verbose:
        print(f"total reduction: {count/(len(im)*len(im[0])):.1%}")

