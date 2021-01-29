import cupy as cp
from utils.vec3 import Vec3, Point3, Vec3List


class Ray:
    def __init__(self, origin = Point3(), direction = Vec3()):
        self.o = origin
        self.d = direction

    def origin(self):
        return self.o

    def direction(self):
        return self.d

    def at(self, t):
        return self.o + self.d * t


class RayList:
    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction

    def origin(self):
        return self.o

    def direction(self):
        return self.d

    def __len__(self):
        return len(self.o)

    def __getitem__(self, idx):
        return Ray(self.o[idx], self.d[idx])

    def __setitem__(self, idx, r):
        self.o[idx] = r.o
        self.d[idx] = r.d

    def __add__(self, r):
        return RayList(self.o + r.o, self.d + r.d)

    def at(self, t):
        return self.o + self.d.mul_ndarray(t)

    @staticmethod
    def single(r):
        return RayList(cp.array([r.o.e]), cp.array([r.d.e]))

    @staticmethod
    def new_empty(length):
        return RayList(Vec3List.new_empty(length), Vec3List.new_empty(length))

    @staticmethod
    def new_zero(length):
        return RayList(Vec3List.new_zero(length), Vec3List.new_zero(length))
