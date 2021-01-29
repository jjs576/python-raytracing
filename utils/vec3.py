import cupy as cp
from utils.utils import random_float, random_float_list


class Vec3:
    def __init__(self, e0 = 0, e1 = 0, e2 = 0):
        self.e: cp.ndarray = cp.array([e0, e1, e2], dtype=cp.float32)

    def x(self):
        return self.e[0]

    def y(self):
        return self.e[1]

    def z(self):
        return self.e[2]

    def __getitem__(self, idx):
        return self.e[idx]

    def __str__(self):
        return f'{self.e[0]} {self.e[1]} {self.e[2]}'

    def length_squared(self):
        return self.e @ self.e

    def length(self):
        return cp.sqrt(self.length_squared())

    def __add__(self, v):
        return Vec3(*(self.e + v.e))

    def __neg__(self):
        return Vec3(*(-self.e))

    def __sub__(self, v):
        return self + (-v)

    def __mul__(self, v):
        if isinstance(v, Vec3):
            return Vec3(*(self.e * v.e))
        return Vec3(*(self.e * v))

    def __matmul__(self, v):
        return self.e @ v.e

    def __truediv__(self, t):
        return self * (1 / t)

    def __iadd__(self, v):
        self.e += v.e
        return self

    def __imul__(self, v):
        if isinstance(v, Vec3):
            self.e *= v.e
        else:
            self.e *= v
        return self

    def __itruediv__(self, t):
        self *= (1 / t)
        return self

    def cross(self, v):
        return Vec3(*cp.cross(self.e, v.e))

    def unit_vector(self):
        length = self.length()
        if length == 0:
            return Vec3()
        return self / length

    def clamp(self, _min, _max):
        return Vec3(*cp.clip(self.e, _min, _max))

    def gamma(self, gamma):
        return Vec3(*(self.e ** (1 / gamma)))

    def reflect(self, n):
        return self - (n * (self @ n)) * 2

    def refract(self, normal, etai_over_etat):
        cos_theta = -self @ normal
        r_out_parallel = (self + normal * cos_theta) * etai_over_etat
        r_out_prep = normal * (-cp.sqrt(1 - r_out_parallel.length_squared()))
        return r_out_parallel + r_out_prep

    @staticmethod
    def random(_min: float = 0, _max: float = 1):
        return Vec3(*random_float_list(3, _min, _max))

    @staticmethod
    def random_in_unit_sphere():
        u = random_float()
        v = random_float()
        theta = u * 2 * cp.pi
        phi = cp.arccos(2 * v - 1)
        r = cp.cbrt(random_float())
        sinTheta = cp.sin(theta)
        cosTheta = cp.cos(theta)
        sinPhi = cp.sin(phi)
        cosPhi = cp.cos(phi)
        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi
        return Vec3(x, y, z)

    @staticmethod
    def random_in_unit_sphere_list(size):
        u = random_float_list(size)
        v = random_float_list(size)
        theta = u * 2 * cp.pi
        phi = cp.arccos(2 * v - 1)
        r = cp.cbrt(random_float_list(size))
        sinTheta = cp.sin(theta)
        cosTheta = cp.cos(theta)
        sinPhi = cp.sin(phi)
        cosPhi = cp.cos(phi)
        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi
        return Vec3List(cp.transpose(cp.array([x, y, z])))

    @staticmethod
    def random_unit_vector(size):
        a = random_float_list(size, 0, 2 * cp.pi)
        z = random_float_list(size, -1, 1)
        r = cp.sqrt(1 - z**2)
        return Vec3List(cp.transpose(cp.array([r*cp.cos(a), r*cp.sin(a), z])))

    @staticmethod
    def random_in_hemisphere(normal):
        in_unit_sphere = Vec3.random_in_unit_sphere_list(len(normal))
        return Vec3List(cp.where(
            in_unit_sphere @ normal > 0, in_unit_sphere.e, -in_unit_sphere.e
        ))

    @staticmethod
    def random_in_unit_disk(size):
        r = cp.sqrt(random_float_list(size))
        theta = random_float_list(size) * 2 * cp.pi
        return cp.array([r*cp.cos(theta), r*cp.sin(theta)])


Point3 = Vec3
Color = Vec3


class Vec3List:
    def __init__(self, e):
        self.e = e

    def x(self):
        return cp.transpose(self.e)[0]

    def y(self):
        return cp.transpose(self.e)[1]

    def z(self):
        return cp.transpose(self.e)[2]

    def cpu(self):
        self.e = cp.asnumpy(self.e)
        return self

    def __getitem__(self, idx):
        return Vec3(*self.e[idx])

    def get_ndarray(self, idx):
        return self.e[idx]

    def __setitem__(self, idx, val):
        self.e[idx] = val.e

    def __len__(self):
        return len(self.e)

    def __add__(self, v):
        return Vec3List(self.e + v.e)

    __radd__ = __add__

    def __iadd__(self, v):
        self.e += v.e
        return self

    def __mul__(self, v):
        if isinstance(v, (Vec3, Vec3List)):
            return Vec3List(self.e * v.e)
        return Vec3List(self.e * v)

    __rmul__ = __mul__

    def __imul__(self, v):
        if isinstance(v, (Vec3, Vec3List)):
            self.e *= v.e
        else:
            self.e *= v
        return self

    def __sub__(self, v):
        return Vec3List(self.e - v.e)

    def __rsub__(self, v):
        return Vec3List(v.e - self.e)

    def __isub__(self, v):
        self.e -= v.e
        return self

    def __neg__(self):
        return Vec3List(-self.e)

    def __truediv__(self, v):
        if isinstance(v, Vec3List):
            return Vec3List(self.e / v.e)
        return Vec3List(self.e / v)

    def __matmul__(self, v):
        return (self.e * v.e).sum(axis=1)

    def mul_ndarray(self, a):
        return self * Vec3List.from_array(a)

    def div_ndarray(self, a):
        return self / Vec3List.from_array(a)

    def as_float32(self):
        self.e = self.e.astype(cp.float32, copy=False)
        return self

    def length_squared(self):
        return (self.e ** 2).sum(axis=1)

    def length(self):
        return cp.sqrt(self.length_squared())

    def unit_vector(self):
        length = self.length()
        condition = length > 0
        length_non_zero = cp.where(condition, length, 1)
        return self.div_ndarray(length_non_zero).mul_ndarray(condition)

    def reflect(self, n):
        return self - (n.mul_ndarray(self @ n)) * 2

    def refract(self, normal, etai_over_etat):
        cos_theta = -self @ normal
        r_out_parallel = (self + normal.mul_ndarray(cos_theta)).mul_ndarray(etai_over_etat)
        r_out_prep = normal.mul_ndarray(-cp.sqrt(1 - r_out_parallel.length_squared()))
        return r_out_parallel + r_out_prep

    @staticmethod
    def from_vec3(v, length):
        v1 = cp.tile(v.e, (length, 1))
        return Vec3List(v1)

    @staticmethod
    def from_array(a):
        vl = cp.transpose(cp.tile(a, (3, 1)))
        return Vec3List(vl)

    @staticmethod
    def new_empty(length):
        return Vec3List(cp.empty((length, 3), dtype=cp.float32))

    @staticmethod
    def new_zero(length):
        return Vec3List(cp.zeros((length, 3), dtype=cp.float32))
