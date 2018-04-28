import cv2
import numbers


class CvResize(object):
    def __init__(self, size, interpolation=cv2.INTER_AREA):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return resize(img, self.size, self.interpolation)

class CvCenterCrop(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        return center_crop(img, self.size)
    

def resize(img, size, interpolation=cv2.INTER_AREA):
    if isinstance(size, int):
        h, w = img.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return cv2.resize(img, (ow, oh), interpolation=interpolation)
    else:
        return cv2.resize(img, size, interpolation=interpolation)
    
def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i+th, j:j+tw]