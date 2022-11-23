import argparse, sys, os, subprocess, string, time
import curses
import numpy as np
import cv2 as cv
import pims

H, W = 13, 7    # height and width of a terminal cell 

def change_size(path: str, h: int, w: int) -> "pims video":
    """ Changes the video size to h x w. """
    # codec requires dimensions be even
    h, w = h - (h % 2) if h > 0 else h, w - (w % 2) if w > 0 else w
    temp = f"scaled_{w}x{h}_{path.split('/')[-1]}"
    # if already scaled, use pre-processed video
    if not os.path.exists(temp):
        subprocess.call(["ffmpeg", "-i", path, "-vf", f"scale={w}:{h},setsar=1:1",
                        "-sws_flags", "bicubic", temp, "-loglevel", "quiet"])
    return pims.open(temp)

def save_video(path: str, video, fps: int) -> None:
    """ Saves a pims video as a mp4 file.
    https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html """
    if os.path.exists(path):
        os.remove(path)
    fourcc = cv.VideoWriter_fourcc(*"avc1")
    out = cv.VideoWriter(path, fourcc, fps, video.frame_shape[:2][::-1])
    for img in video:
        out.write(cv.cvtColor(255*img.astype(np.uint8), cv.COLOR_GRAY2RGB))
    out.release()

def play_audio(path: str) -> subprocess.Popen:
    """ Plays the audio file without blocking. """
    return subprocess.Popen(["afplay", path])

def histogram(img: np.array, norm: bool=False) -> np.array:
    """ Computes the distribution of the image. """
    p = np.zeros(256, dtype=np.int)
    for x in img.flatten():
        p[x] += 1
    return p if not norm else p/np.sum(p), np.sum(p)

def __threshold(img: np.array, threshold: int) -> np.array:
    """ Binarizes an image given a threshold. """
    return img > threshold

@pims.pipeline
def otsu(img: np.array) -> np.array:
    """ Applies Otsu's binarization to the image with intra-class variance. """
    # if only one pixel value, variance is constant regardless of threshold
    if len(set(img.flatten())) == 1:
        return __threshold(img, 127)
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

@pims.pipeline
def otsu_cv(img: np.array) -> np.array:
    """ Applies Otsu's binarization using openCV's implementation.
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html """
    # img = cv.GaussianBlur(img, (5, 5), 0) # reduce noise
    return cv.threshold(img, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)[1] > 0

@pims.pipeline
def scale(img: np.array) -> np.array:
    """ Converts a float64 image to a uint8 image. """
    return np.clip(img, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terminal video renderer.")
    parser.add_argument("-v", "--version", action="version", version="1.0")
    parser.add_argument("video", help="path to video file")
    parser.add_argument("-a", "--audio", help="path to audio file")
    parser.add_argument("-t", "--text", help="path to text file")
    parser.add_argument("-H", "--height", type=int,
                        help="height, terminal height by default")
    parser.add_argument("-w", "--width", type=int,
                        help="width, terminal width by default")
    parser.add_argument("-r", "--rate", type=float, default=30,
                        help="frame rate: pass 0 for input video rate")
    parser.add_argument("-m", "--mode", default="light",
                        choices=["light", "dark"], help="background color")
    parser.add_argument("-d", "--display", default="print",
                        choices=["print", "write"], help="display method")

    args = parser.parse_args()

    light_mode = args.mode == "light"
    fps = video.frame_rate if args.rate == 0 else args.rate
    video = pims.open(args.video)
    # account for aspect ratio of terminal cells
    ar = (video.frame_shape[1]/video.frame_shape[0])*(H/W)

    # scale video down to proper size
    try:
        term_width, term_height = os.get_terminal_size()
    except OSError:
        term_width, term_height = 0, 0
    term_height -= args.display == "write" # one less useable line in vim
    h, w = args.height, args.width
    if h == 0: h = term_height
    if w == 0: w = term_width
    if h is not None or w is not None: # user provided size
        h, w = (h, h*ar) if h is not None else (w/ar, w)
    else:                              # default to best fit
        h, w = (term_height, term_height*ar) if term_height*ar < term_width \
                else (term_width/ar, term_width)
    h, w = int(h), int(w)
    if h > term_height or w > term_width:
        print(f"Invalid dimensions: {h}x{w}, must be less than {term_height}x{term_width}")
        exit()

    video = otsu_cv(scale(pims.as_grey(change_size(args.video, h, w))))
    # visualize resulting video
    # save_video(f"binary_{args.path}", video, fps)

    # load text
    N = np.prod(video.frame_shape[:2])
    if args.text is not None:
        with open(args.text) as f:
            whitespace = set(string.whitespace)
            s = "".join(ch for ch in f.read() if ch not in whitespace)
    else:
        s = "@"*N
    if len(s) < N:
        print(f"Not enough characters in {args.text}: {len(s)}, need {N}")
        exit()
    s = np.array(list(s))

    # play music
    if args.audio is not None:
        music = play_audio(args.audio)

    # kill music on error
    def except_hook(exctype, value, traceback):
        if args.audio is not None:
            music.kill()
        sys.__excepthook__(exctype, value, traceback)
    sys.excepthook = except_hook

    def main(stdscr: curses.window) -> None:
        """ Main loop wrapped with curses. """
        # curses setup
        if args.display == "print":
            curses.init_color(curses.COLOR_WHITE, 1000, 1000, 1000)
            # set background
            if light_mode:
                curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
                stdscr.bkgd(" ", curses.color_pair(1))
            # turn off cursor
            curses.curs_set(0)
            stdscr.clear()

        start = time.time()
        for i, frame in enumerate(video):
            # block until right time
            while time.time() - start < i/fps:
                pass
            frame ^= not light_mode
            # form string matrix
            t = np.full(frame.shape, " ", dtype=(np.unicode_, 1))
            t[~frame] = s[:(~frame).sum()]
            t = "\n".join(t.view((np.unicode_, frame.shape[1])).ravel())
            # display to terminal
            if args.display == "print":
                stdscr.erase()
                stdscr.addstr(t)
                stdscr.refresh()
            elif args.display == "write":
                with open("out.txt", "w") as f:
                    f.write(t + "\n")

        # end music
        if args.audio is not None:
            music.kill()

    # start main loop
    curses.wrapper(main) if args.display == "print" else main(None)

