from PIL import ImageGrab
import numpy as np
image = ImageGrab.grab()
print("屏幕", image.size)
image.save(r"F:\born-wired-cortex\engine_v2\artifacts\_桌面.png")
print("saved artifacts/_桌面.png")