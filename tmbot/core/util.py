import numpy as np
import win32gui
import win32ui
import win32con
import os
import pyautogui
from pathlib import Path
from PIL import Image

def get_frame(shape : tuple[int, int], mode : str = "L", crop : bool = False, algorithm : str = "pywinauto"):
    assert algorithm in ["pywinauto", "win32"]

    if algorithm == "pywinauto":
        frame = pyautogui.screenshot()
    elif algorithm == "win32":
        frame = _screenshot()

    if crop:
        frame = _square_crop(frame)

    return np.asarray(frame.convert(mode).resize(shape), dtype=np.uint8).transpose()

def _screenshot():
    hwnd = win32gui.FindWindow(None, "Trackmania")

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    dc = mfcDC.CreateCompatibleDC()

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfcDC, w, h)
    dc.SelectObject(bmp)

    dc.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
    buffer = bmp.GetBitmapBits(True)

    dc.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    win32gui.DeleteObject(bmp.GetHandle())

    return Image.frombuffer("RGBA", (w, h), buffer)

def _square_crop(image : Image.Image):
    width, height = image.size
    offset  = int(abs(height-width)/2)
    if width > height:
        image = image.crop([offset, 0, width-offset, height])
    elif width < height:
        image = image.crop([0, offset, width, height-offset])
    return image

def get_default_op_path():
    return Path(os.path.expanduser("~"), "OpenPlanetNext")

def linear_interp(value : float, end : float, intercept : float = 0):
    """Linearly interpolates between 0 and end with an intercept."""
    return ((value - 1) / (end - 1)) * (1 - intercept) + intercept

# Conversion util functions

def norm_float(value : float, min : float, max : float) -> float:
    value = float(value)
    return ((value - min) / (max - min)) * 2 - 1 # Scale value proportionally between min and max

def binary_strbool(value : str) -> int:
    if value == "true":
        return 1
    elif value == "false":
        return 0
    raise ValueError

# None, tech, plastic, dirt, grass, ice, other
#   0     1      2      3      4     5     6
mat_idx = {
    "Asphalt" : 1,
    "Concrete" : 1,
    "Pavement" : 1,
    "Grass" : 4,
    "Ice" : 5,
    "Metal" : 1,
    "Sand" : 4,
    "Dirt" : 3,
    "DirtRoad" : 3,
    "Rubber" : 2,
    "SlidingRubber" : 2,
    "Water" : 2,
    "WetDirtRoad" : 3,
    "WetAsphalt" : 1,
    "WetPavement" : 1,
    "WetGrass" : 4,
    "Snow" : 4,
    "Tech" : 1,
    "TechHook" : 1,
    "TechGround" : 1,
    "TechWall" : 1,
    "TechArrow" : 1,
    "RoadIce" : 5,
    "Green" : 4,
    "Plastic" : 2,
    "XXX_Null" : 0,
}

def mat_index(value : str) -> int:
    try:
        return mat_idx[value]
    except:
        print(value, "is not an assigned surface")
        return 6 # Other

# None, Playing, Finish
#   0      1       2
race_idx = {
    "Playing" : 1,
    "Finish" : 2,
}

def race_index(value : str) -> int:
    try:
        return race_idx[value]
    except:
        return 0 # Other
