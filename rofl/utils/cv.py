import cv2
from rofl.functions.const import FRAME_SIZE

def imgResize(f, size = FRAME_SIZE):
    return cv2.resize(f, size)

def YChannelResize(f, size = FRAME_SIZE):
    f = cv2.cvtColor(f, cv2.COLOR_RGB2YUV)[:,:,0]
    return imgResize(f, size)