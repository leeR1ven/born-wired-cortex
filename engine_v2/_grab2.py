from PIL import ImageGrab
image = ImageGrab.grab()
image.save(r"F:\born-wired-cortex\engine_v2\artifacts\_桌面2.png")
print("saved", image.size)