import numpy as np
from PIL import Image
from utils.vec3 import Color, Vec3List


class Img:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.frame = np.zeros((h, w, 3), dtype=np.float64)

    def set_frame(self, array):
        self.frame = array

    def write_pixel(self, w, h, pixel_color, samples_per_pixel):
        color = pixel_color / samples_per_pixel
        self.frame[h][w] = color.clamp(0, 0.999).gamma(2).e

    def write_pixel_list(self, h, pixel_color_list, samples_per_pixel):
        color = pixel_color_list.e / samples_per_pixel
        gamma = 2
        self.frame[h] = np.clip(color, 0, 0.999) ** (1 / gamma)

    def write_frame(self, frame):
        self.frame += frame.e.reshape((self.h, self.w, 3))
        return self

    def average(self, samples_per_pixel):
        self.frame /= samples_per_pixel
        return self

    def gamma(self, gamma):
        self.frame = np.clip(self.frame, 0, 0.999) ** (1 / gamma)
        return self

    def save(self, path, show = False):
        im = Image.fromarray(np.uint8(self.frame * 255))
        im.save(path)
        if show:
            im.show()
