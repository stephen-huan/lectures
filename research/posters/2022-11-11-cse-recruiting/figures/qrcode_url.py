import qrcode
import qrcode.image.svg
import qrcode.image.styles.colormasks
from qrcode.image.styles.colormasks import SolidFillColorMask
# https://github.com/lincolnloop/python-qrcode

# url to project website
DATA = "https://stephen-huan.github.io/projects/cholesky/"
# FILL_COLOR = "black"
FILL_COLOR = (121, 135, 149)
BACK_COLOR = "white"

qr = qrcode.QRCode(
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    # image_factory=qrcode.image.svg.SvgPathImage,
)
qr.add_data(DATA)
qr.make(fit=True)

img = qr.make_image(fill_color=FILL_COLOR, back_color=BACK_COLOR)
img.save("qrcode.png")

