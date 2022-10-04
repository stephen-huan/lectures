import math, time
from functools import lru_cache
from collections import deque

### common mathematical functions
atan2, exp, ln = math.atan2, math.exp, math.log
# assume negatives are floating-point error
sqrt = lambda x: math.sqrt(x) if x >= 0 else 0
sign = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
# sue me but all the formulas are in degrees and I don't feel like converting
pi, deg, rad = math.pi, math.degrees, math.radians
sin, cos = lambda x: math.sin(rad(x)), lambda x: math.cos(rad(x))

def to_base(n: int, b: int) -> list:
    """ converts a number from base 10 to base b. """
    l = []
    while n != 0:
        l.append(n % b)
        n //= b
    return l[::-1]

def from_base(n: int, b: int) -> int:
    """ Converts a number from base b to base 10. """
    rtn, e = 0, 1
    while n != 0:
        rtn += (n % b)*e
        n //= 10
        e *= b
    return rtn

### matrix operations

def tmap(f, x) -> tuple:
    """ Apply a function on a vector. """
    return tuple(map(f, x))

def scale(x: tuple, f: float) -> tuple:
    """ Multiply a vector by a value. """
    return tmap(lambda v: v*f, x)

def column(u: list) -> list:
    """ Converts a normal python list into a column vector. """
    return [[x] for x in u]

def dotp(u: list, v: list) -> float:
    """ hadamard product """
    return tuple(x*y for x, y in zip(u, v))

def dot(u: list, v: list) -> float:
    """ dot product """
    return sum(dotp(u, v))

def norm(u: tuple) -> float:
    """ Magnitude of a vector. """
    return sqrt(dot(u, u))

def flatten(m: list) -> list:
    """ Flattens a matrix into a vector.  """
    return tuple(x for row in m for x in row)

def diagonal(l: list) -> list:
    """ Turns a list into a diagonal matrix. """
    return [[(i == j)*l[i] for j in range(len(l))] for i in range(len(l))]

def col(m: list, i: int) -> list:
    """ column of a matrix """
    return [m[j][i] for j in range(len(m))]

def mult(a: list, b: list) -> list:
    """ matrix-matrix multiplication """
    return [[dot(a[i], col(b, j)) \
             for j in range(len(b[0]))] for i in range(len(a))]

def mulv(m: list, v: list) -> list:
    """ matrix-vector multiplication """
    return tuple(dot(row, v) for row in m)

def inv(m):
    """ Inverts a matrix. Not exactly trustworthy, I'd use np.linalg.inv """
    N = len(m)
    # augment matrix with identity
    for i in range(N):
        m[i] += [i == j for j in range(N)]

    for c in range(N):
        # get first nonzero entry  
        pivot = [i for i in range(c, N) if m[i][c] != 0][0]
        m[c], m[pivot] = m[pivot], m[c]
        v = m[c][c]
        # set pivot value to 1
        for i in range(len(m[c])):
            m[c][i] /= v
        # make all zeros
        for r in range(N):
            if r != c:
                v = m[r][c]
                for i in range(len(m[r])):
                    m[r][i] -= v*m[c][i]

    return [row[N:] for row in m]

### conversion functions for ANSI color codes

def esc_color(s: str) -> tuple:
    """ Turns a color code into a tuple. """
    return tmap(int, s[2:-1].split(";")[2::3])

def color_esc(c: tuple) -> str:
    """ Turns a tuple into a color code. """
    c = (38, 5, c[0], 48, 5, c[1]) if len(c) == 2 else (38, 5, c[0])
    return f"\x1b[{';'.join(map(str, c))}m"

# taken from https://github.com/Qix-/color-convert/blob/master/conversions.js

def ansi256_rgb(c: int) -> tuple:
    """ Turns an ansi color code into a rgb tuple. """
    if c >= 232: # greyscale
        return scale((10*(c - 232) + 8,)*3, 1/255)
    digits = to_base(c - 16, 6) # write in base 6, pad with 0's
    return scale([0]*(3 - len(digits)) + digits, 1/5)

def rgb_ansi256(c: tuple) -> int:
    """ Turns a rgb tuple into an ansi color code. """
    r, g, b = scale(c, 255)
    if r == g and g == b: # greyscale
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + round(24*(r - 8)/247)
    return 16 + int("".join(map(lambda x: str(round(5*x)), c)), 6)

# or just use https://github.com/magarcia/python-x256
# same list is provided here: https://jonasjacek.github.io/colors/
# xterm-256 colors differ signifcantly from the ANSI standard... 
from x256 import colors as ansi_colors

def ansi256_srgb(c: int) -> tuple:
    """ Uses the table from python-x256. """
    return ansi_colors[c]

def srgb_ansi256(c: tuple) -> int:
    """ sRGB color to ansi color code. """
    return rgb_ansi256(scale(c, 1/255))
    return ansi_colors.index(c)

### sRGB color space vs "linear" RGB: https://en.wikipedia.org/wiki/SRGB

def rgb_srgb(c: tuple) -> tuple:
    """ Linear RGB to sRGB. """
    f = lambda x: 1.055*(x**(1/2.4)) - 0.055 if x > 0.0031308 else 12.92*x
    return tmap(lambda x: round(255*f(x)), c)

def srgb_rgb(c: tuple) -> tuple:
    """ sRGB to linear RGB. """
    c = scale(c, 1/255) # sRGB in [0, 255] and linear RGB in [0, 1]
    finv = lambda x: ((x + 0.055)/1.055)**2.4 if x > 0.04045 else x/12.92
    return tmap(finv, c)

### YUV color space: https://github.com/posva/catimg/blob/master/src/sh_color.c
# https://en.wikipedia.org/wiki/YUV
rgb2yuv = [[ 0.299  ,  0.587  ,  0.114  ],
           [-0.14173, -0.28886,  0.436  ],
           [ 0.615  , -0.51499, -0.10001]]
yuv2rgb = [[ 1,  0      ,  1.13983],        # recommended but a poor inverse
           [ 1, -0.39465, -0.58060],
           [ 1,  2.03211,  0      ]]
yuv2rgb = [[ 1.00000006, -0.0000118 ,  1.13983465], # np.linalg.inv(rgb2yuv)
           [ 1.00213504, -0.39464608, -0.57816515],
           [ 0.98900627,  2.03211207, -0.01252298]]

def rgb_yuv(c: tuple) -> tuple:
    """ Linear RGB to YUV color space. """
    # this is what catimg does, and scikit-image for that matter
    # why is YUV on sRGB such a good perceptually uniform color space?
    return mulv(rgb2yuv, scale(rgb_srgb(c), 1/255))
    return mulv(rgb2yuv, c)

def yuv_rgb(c: tuple) -> tuple:
    """ YUV to linear RGB color space. """
    return srgb_rgb(tmap(lambda x: 255*min(max(x, 0), 1), mulv(yuv2rgb, c)))
    return tmap(lambda x: min(max(x, 0), 1), mulv(yuv2rgb, c))

### xyz color space
"""
RGB -> XYZ: http://www.easyrgb.com/en/math.php
"""
rgb2xyz = [[ 0.412453,  0.357580,  0.180423],
           [ 0.212671,  0.715160,  0.072169],
           [ 0.019334,  0.119193,  0.950227]]
# xyz2rgb = np.linalg.inv(rgb2xyz)
xyz2rgb = [[ 3.24048134, -1.53715152, -0.49853633],
           [-0.96925495,  1.87599   ,  0.04155593],
           [ 0.05564664, -0.20404134,  1.05731107]]

def rgb_xyz(c: tuple) -> tuple:
    """ Linear RGB to XYZ color space. """
    return scale(mulv(rgb2xyz, c), 100)

def xyz_rgb(c: tuple) -> tuple:
    """ XYZ to linear RGB color space. """
    return mulv(xyz2rgb, scale(c, 1/100))

### lab color space
"""
XYZ -> LAB: https://en.wikipedia.org/wiki/CIELAB_color_space
just use: https://scikit-image.org/docs/dev/api/skimage.color.html#rgb2lab
"""
# Standard Illuminant D65 - https://en.wikipedia.org/wiki/Illuminant_D65
REF = [95.047, 100, 108.883]

def xyz_lab(c: tuple) -> tuple:
    """ XYZ to LAB color space. """
    # constants
    f = lambda x: x**(1/3) if x > (6/29)**3 else x/(3*(6/29)**2) + 4/29
    x, y, z = map(f, (n/d for n, d in zip(c, REF)))
    return (116*y - 16, 500*(x - y), 200*(y - z))

def lab_xyz(c: tuple) -> tuple:
    """ LAB color space to XYZ. """
    y = (c[0] + 16)/116
    f = lambda x: x**3 if x > 6/29 else 3*((6/29)**2)*(x - 4/29)
    return dotp(REF, map(f, (y + c[1]/500, y, y - c[2]/200)))

### ciecam02, ciecam02ucs, cam16, and cam16ucs color space
"""
see "The CIECAM02 color appearance model"
https://www.researchgate.net/publication/221501922_The_CIECAM02_color_appearance_model
and "Uniform Colour Spaces Based on CIECAM02 Colour Appearance Model"
https://doi.org/10.1002/col.20227

see "Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS",
https://doi.org/10.1002/col.22131
and "Algorithmic improvements for the CIECAM02 and CAM16 color appearance models",
https://arxiv.org/abs/1802.06067
"""
M02 = [[ 0.7328,  0.4296, -0.1624],                # page 3, equation 7 
       [-0.7036,  1.6975,  0.0061],
       [ 0.0030,  0.0136,  0.9834]]
M02inv = [[ 1.096124, -0.278869,  0.182745],       # page 3, equation 11
          [ 0.454369,  0.473533,  0.072098],
          [-0.009628, -0.005698,  1.015326]]
MH = [[ 0.38971,  0.68898, -0.07868],              # page 3, equation 12
      [-0.22981,  1.18340,  0.04641],
      [       0,        0,        1]]

M16 = [[ 0.401288,  0.650173, -0.051461],          # page 4, equation 18
       [-0.250268,  1.204414,  0.045854],
       [-0.002079,  0.048952,  0.953127]]
# apparently not equal to np.linalg.inv(M16) but I can't tell the difference
M16inv = [[ 1.86206786, -1.01125463,  0.14918677], # page 15 
          [ 0.38752654,  0.62144744, -0.00897398],
          [-0.01584150, -0.03412294,  1.04996444]]


Mdot = [[2, 1, 1/20], [1, -12/11, 1/11], [1/9, 1/9, -2/9], [1, 1, 21/20]]
Mdotinv = [[460, 451, 288], [460, -891, -261], [460, -220, -6300]]

hue_data = (( 20.14, 0.8,   0.0),                  # page 14, table A2
            ( 90.00, 0.7, 100.0),
            (164.25, 1.0, 200.0),
            (237.53, 1.2, 300.0),
            (380.14, 0.8, 400.0))

Yw = REF[1] # reference white

def hue_angle(b: float, a: float) -> float:
    """ Computes the hue angle carefully. """
    return deg(atan2(b, a)) % 360

class CAM:

    """ Class so I don't pollute global with 20 billion parameters. """

    # see https://github.com/nschloe/colorio for Yb and Ew
    def __init__(self, c: float, Yb: float, Ew: float,
                 m1: list, m2: list=diagonal((1,)*3)) -> None:
        F, self.c, self.Nc = 1.0, c, 1.0 # page 13, table A1
        # Ew = illuminance of reference white in lux, see equation A1
        LA = (Ew/pi)*(Yb/Yw)

        # step 0: claculate all values independent of the input sample
        RGBw = mulv(m1, REF)
        D = min(max(F*(1 - (1/3.6)*exp((-LA - 42)/92)), 0), 1)
        self.DC = tuple(D*Yw/x + 1 - D for x in RGBw)
        self.MDC = diagonal(self.DC)

        k = 1/(5*LA + 1)
        self.FL = 0.2*k**4*(5*LA) + 0.1*(1 - k**4)**2*(5*LA)**(1/3)

        self.n = Yb/Yw
        self.z, self.Nbb = 1.48 + sqrt(self.n), 0.725*(1/self.n)**0.2
        self.Ncb = self.Nbb

        RGBwc = mulv(m2, dotp(self.DC, RGBw))
        self.post = lambda x, f=self.FL: \
            400*(f*x/100)**0.42/((f*x/100)**0.42 + 27.13)
        self.postinv = lambda x, f=self.FL: \
            sign(x)*100/f*(27.13*abs(x)/(400 - abs(x)))**(1/0.42)
        RGBaw = tmap(self.post, RGBwc)
        self.Aw = dot(Mdot[0], RGBaw)*self.Nbb

    def CAT(self, c: tuple) -> tuple:
        """ Linear transformation. """
        # combines step 1: cone responses and step 2: color adaptation
        return mulv(self.M, c)

    def CATinv(self, c: tuple) -> tuple:
        """ Undo the color appearance transformation. """
        # combines step 6: RGB and step 7: X, Y, Z
        return mulv(self.Minv, c)

    def CAM(self, c: tuple) -> tuple:
        """ Color appearance model after color appearance transform. """
        # step 3: postadaptation cone response
        RGBp = tmap(lambda x: sign(x)*self.post(abs(x)), c)
        # step 4: color components a/b, hue angle h, auxililary variables pp2/u
        pp2, a, b, u = mulv(Mdot, RGBp)
        h = hue_angle(b, a)
        # step 5: eccentricity
        hp = h + (h < hue_data[0][0])*360
        for i in range(len(hue_data) - 1):
            if hue_data[i][0] <= hp < hue_data[i + 1][0]:
                break
        et = 1/4*(cos(hp + deg(2)) + 3.8)
        hi, ei, Hi, h1, e1, H1 = hue_data[i] + hue_data[i + 1]
        H = Hi + 100*e1*(hp - hi)/(e1*(hp - hi) + ei*(h1 - hp))
        # PL, PR = round(H1 - H), round(H - Hi)
        A = pp2*self.Nbb                     # step 6: achromatic response 
        J = 100*(A/self.Aw)**(self.c*self.z) # step 7: correlate of lightness
        # step 8: correlate of brightness
        Q = 4/self.c*sqrt(J/100)*(self.Aw + 4)*self.FL**0.25
        # step 9: correlate of chroma C, colorfulness M, saturation s
        t = 50000/13*self.Nc*self.Ncb*et*sqrt(a**2 + b**2)/(u + 0.305)
        alpha = t**0.9*(1.64 - 0.29**self.n)**0.73
        C = alpha*sqrt(J/100)
        M = C*self.FL**0.25
        s = 50*sqrt(alpha*self.c/(self.Aw + 4))
        return (J, C, H, h, M, s, Q)

    def CAMinv(self, c: tuple) -> tuple:
        """ Reverse model of the color appearance transform. """
        J, _, _, h, M, _, _ = c
        # step 1: get J, t, and h
        C = M/self.FL**0.25
        alpha = 0 if J == 0 else C/(sqrt(J/100))
        t = (alpha/(1.64 - 0.29**self.n)**0.73)**(1/0.9)
        # step 2: et, A, pp1, pp2 
        et = 1/4*(cos(h + deg(2)) + 3.8)
        A = self.Aw*(J/100)**(1/(self.c*self.z))
        pp1 = et*50000/13*self.Nc*self.Ncb
        pp2 = A/self.Nbb
        # step 3: a and b
        gamma = 23*(pp2 + 0.305)*t/(23*pp1 + 11*t*cos(h) + 108*t*sin(h))
        a, b = gamma*cos(h), gamma*sin(h)
        RGBp = scale(mulv(Mdotinv, (pp2, a, b)), 1/1403) # step 4: RGBp
        RGBc = tmap(self.postinv, RGBp)                  # step 5: RGBc
        return RGBc

    def from_xyz(self, c: tuple) -> tuple:
        """ XYZ to CAMXY color space. """
        return self.CAM(self.CAT(c))

    def to_xyz(self, c: tuple) -> tuple:
        """ CAMXY to XYZ color space. """
        return self.CATinv(self.CAMinv(c))

    # see "Uniform colour spaces based on CIECAMO2 colour appearance model",
    # https://doi.org/10.1002/col.20227
    def to_ucs(self, c: tuple, c1: float=0.007, c2: float=0.0228) -> tuple:
        """ CAMXY to CAMXY-UCS color space. """
        J, _, _, h, M, _, _ = c
        Jp, Mp = (1 + 100*c1)*J/(1 + c1*J), ln(1 + c2*M)/c2
        return (Jp, Mp*cos(h), Mp*sin(h))

    def from_ucs(self, c: tuple, c1: float=0.007, c2: float=0.0228) -> tuple:
        """ CAMXY-UCS to CAMXY color space. """
        Jp, ap, bp = c
        Mp, h = sqrt(ap**2 + bp**2), hue_angle(bp, ap)
        M, J = (exp(c2*Mp) - 1)/c2, Jp/(1 + c1*(100 - Jp))
        return (J, None, None, h, M, None, None)

# both use the same base model, just different initial transform
# for UCS derivative, both use the same parameters

class CAM02(CAM):

    """ CIECAM02 color apperance model. """

    def __init__(self, c: float=0.69, Yb: float=20, Ew: float=64) -> None:
        super().__init__(c, Yb, Ew, M02, mult(MH, M02inv))
        self.M = mult(MH, mult(M02inv, mult(self.MDC, M02)))
        self.Minv = [[ 1.8434534 , -1.10174947,  0.21654439],
                     [ 0.33096753,  0.66524799,  0.0034579 ],
                     [ 0.00154042, -0.00015316,  1.07142775]]

class CAM16(CAM):

    """ CAM16 color apperance model. """

    def __init__(self, c: float=0.69, Yb: float=20, Ew: float=64) -> None:
        super().__init__(c, Yb, Ew, M16)
        self.M = mult(self.MDC, M16)
        self.Minv = [[ 1.82405269, -1.02506701,  0.15955629],
                     [ 0.37961497,  0.62993558, -0.00959774],
                     [-0.01551809, -0.03458901,  1.12294422]]

cam02, cam16 = CAM02(), CAM16()

### general conversions

def bfs(graph: dict, start: str) -> dict:
    """ Finds the shortest path (least functions) to each other color. """
    funcs = {start: start}
    q = deque([start])
    while len(q) > 0:
        n = q.popleft()
        for child in graph[n]:
            if child not in funcs:
                funcs[child] = n
                q.append(child)
    return funcs

# graph[source][target] for conversion from source color to target color
graph = {"ansi256": {"srgb": ansi256_srgb},
         "srgb": {"ansi256": srgb_ansi256, "rgb": srgb_rgb},
         "rgb": {"srgb": rgb_srgb, "yuv": rgb_yuv, "xyz": rgb_xyz},
         "yuv": {"rgb": yuv_rgb},
         "xyz": {"rgb": xyz_rgb, "lab": xyz_lab,
                 "cam02": cam02.from_xyz, "cam16": cam16.from_xyz},
         "lab": {"xyz": lab_xyz},
         "cam02": {"xyz": cam02.to_xyz, "cam02ucs": cam02.to_ucs},
         "cam02ucs": {"cam02": cam02.from_ucs},
         "cam16": {"xyz": cam16.to_xyz, "cam16ucs": cam16.to_ucs},
         "cam16ucs": {"cam16": cam16.from_ucs},
        }
new_graph = {color: bfs(graph, color) for color in graph}

@lru_cache(maxsize=None)
def convert(c: tuple, source: str, target: str) -> tuple:
    """ Converts a color from the source space to the target space. """
    order = [target]
    while order[-1] != source:
        order.append(new_graph[source][order[-1]])
    curr = source
    for child in order[:-1][::-1]:
        c = graph[curr][child](c)
        curr = child
    return c if target != "ansi256" else (c,)

conv = {color1: {color2: lambda c, c1=color1, c2=color2: convert(c, c1, c2)
                 for color2 in graph} for color1 in graph}

### distance metrics

def dist(u: tuple, v: tuple) -> float:
    """ Distance between two vectors. """
    return sqrt(sum((x - y)*(x - y) for x, y in zip(u, v)))

# constants determined by lighting and application
kL, kC, kH, K1, K2 = 1, 1, 1, 0.045, 0.015

# http://colormine.org/delta-e-calculator
# https://en.wikipedia.org/wiki/Color_difference
def cie94(u: tuple, v: tuple) -> float:
    """ 1994 CIE color difference in the CIELAB space."""
    dL, da, db = (u[i] - v[i] for i in range(3))
    C1, C2 = norm(u[1:]), norm(v[1:])
    dC = C1 - C2
    dH = sqrt(da**2 + db**2 - dC**2)
    SL, SC, SH = 1, 1 + K1*C1, 1 + K2*C1
    dE = sum((n/d)**2 for n, d in zip((dL, dC, dH), (kL*SL, kC*SC, kH*SH)))
    return sqrt(dE)

# http://www2.ece.rochester.edu/~gsharma/ciede2000/
def cie2000(u: tuple, v: tuple) -> float:
    """ 2000 CIE color difference in the CIELAB space."""
    # part 1: Cp and hp 
    C1, C2 = norm(u[1:]), norm(v[1:])
    Cavg = (C1 + C2)/2
    f = lambda x: sqrt(x**7/(x**7 + 25**7)) # weird function that's used again
    G = 0.5*(1 - f(Cavg))
    ap1, ap2 = (1 + G)*u[1], (1 + G)*v[1]
    x1, x2 = (ap1, u[2]), (ap2, v[2])
    Cp1, Cp2 = norm(x1), norm(x2)
    hp1, hp2 = (hue_angle(b, a) for a, b in (x1, x2))
    # part 2: delta Lp, delta Cp, delta Hp 
    dLp, dCp = v[0] - u[0], Cp2 - Cp1
    CC = Cp1*Cp2
    sub, add = hp2 - hp1, hp2 + hp1
    dhp = (CC != 0)*(sub + (abs(sub) > 180)*360*(-1)**(sub > 180))
    dHp = 2*sqrt(CC)*sin(dhp/2)
    # part 3: compute distance
    Lavgp, Cavgp = (u[0] + v[0])/2, (Cp1 + Cp2)/2
    havgp = add/2 + (abs(sub) > 180)*180*(-1)**(add >= 360) if CC != 0 else add
    T = 1 - 0.17*cos(havgp - 30)  + 0.24*cos(2*havgp) \
          + 0.32*cos(3*havgp + 6) - 0.20*cos(4*havgp - 63)
    dtheta = 30*exp(-((havgp - 275)/25)**2)
    Rc = 2*f(Cavgp)
    SL = 1 + K2*(Lavgp - 50)**2/sqrt(20 + (Lavgp - 50)**2)
    SC = 1 + K1*Cavgp
    SH = 1 + K2*Cavgp*T
    RT = -sin(2*dtheta)*Rc
    fL, fC, fH = (n/d for n, d in zip((dLp, dCp, dHp), (kL*SL, kC*SC, kH*SH)))
    deltaE00 = fL**2 + fC**2 + fH**2 + RT*fC*fH
    return sqrt(deltaE00)

# http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CMC.html
def cmc(u: tuple, v: tuple, l: float=2, c: float=1) -> float:
    """ CMC color difference - note: not symmetric. """
    dL, da, db = (u[i] - v[i] for i in range(3))
    C1, C2 = norm(u[1:]), norm(v[1:])
    dC = C1 - C2
    dH = sqrt(da**2 + db**2 - dC**2)

    h1 = hue_angle(u[2], u[1])
    T = 0.56 + abs(0.2*cos(h1 + 168)) if 164 <= h1 <= 345 else \
        0.36 + abs(0.4*cos(h1 +  35))
    F = sqrt(C1**4/(C1**4 + 1900))

    SL = 0.511 if u[0] < 16 else 0.040975*u[0]/(1 + 0.01765*u[0])
    SC = 0.0638*C1/(1 + 0.0131*C1) + 0.638
    SH = SC*(F*T + 1 - F)
    dE = sum((n/d)**2 for n, d in zip((dL, dC, dH), (l*SL, c*SC, SH)))
    return sqrt(dE)

metrics = {"euclidean": dist,
           "cie76": dist,
           "cie94": cie94,
           "cie2000": cie2000,
           "cmc": cmc,
          }

# see "Power functions improving the performance of color-difference formulas",
# https://doi.org/10.1364/OE.23.000597
powers = {"cie76": (1.26, 0.55),
          "cie94": (1.41, 0.70),
          "cie2000": (1.43, 0.70),
          "cmc": (1.34, 0.66),
          "lab": (1.26, 0.55),
          "cam02ucs": (1.30, 0.75),
          "cam16ucs": (1.41, 0.63),
         }

def power(space: str, metric: str):
    """ Returns the metric transformed by the power function. """
    a, b = powers[metric] if metric in powers else powers[space]
    return lambda u, v, a=a, b=b: a*metrics[metric](u, v)**b

### careful conversion to ansi
# kinda expensive
spaces = {color: {i: conv["ansi256"][color](i) for i in range(16, 256)}
          for color in graph}

@lru_cache(maxsize=None)
def rectify(c: tuple, space: str="lab", metric=metrics["euclidean"]) -> int:
    """ Finds the closest ansi color to the given color,
        being careful to only pick possible ansi colors. """
    return min(range(16, 256), key=lambda i: metric(c, spaces[space][i]))

if __name__ == "__main__":
    EPSILON = 5*10**-5 # 4 digits
    # test ciede2000 implementation
    with open("ciede2000testdata.txt") as f:
        lines = [list(map(float, line.split())) for line in f][:-1]
        for i, row in enumerate(lines):
            c1, c2, d = row[:3], row[3:-1], row[-1]
            d1, d2 = cie2000(c1, c2), cie2000(c2, c1)
            assert abs(d1 - d2) <= EPSILON, f"not symmetric: {d1}, {d2}"
            assert abs(d1 - d) <= EPSILON, f"{i + 1}: {c1} {c2} => {d}, said {d1}"

    print("all tests passed!")

