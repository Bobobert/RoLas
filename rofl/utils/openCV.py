import cv2
from rofl.functions.const import FRAME_SIZE

def imgResize(f, size = FRAME_SIZE):
    if f.shape[0] <= FRAME_SIZE[0] and f.shape[1] <= FRAME_SIZE[1]:
        return f
    return cv2.resize(f, size)

def YChannelResize(f, size = FRAME_SIZE):
    f = cv2.cvtColor(f, cv2.COLOR_RGB2YUV)[:,:,0]
    return imgResize(f, size)