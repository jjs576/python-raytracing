import cupy as cp
from utils.vec3 import Vec3, Point3, Vec3List
from utils.ray import RayList
from utils.utils import degrees_to_radians


class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist):
        theta = degrees_to_radians(vfov)
        h = cp.tan(theta / 2)
        viewport_height = 2 * h
        viewport_width = aspect_ratio * viewport_height

        self.w = (lookfrom - lookat).unit_vector()
        self.u = vup.cross(self.w).unit_vector()
        self.v = self.w.cross(self.u)

        self.origin = lookfrom
        self.horizontal = self.u * viewport_width * focus_dist
        self.vertical = self.v * viewport_height * focus_dist
        self.top_left_corner = self.origin - self.horizontal / 2 + self.vertical / 2 - self.w * focus_dist
        self.lens_radius = aperture / 2

    def get_ray(self, s, t):
        if len(s) != len(t):
            raise ValueError
        rd = Vec3.random_in_unit_disk(len(s)) * self.lens_radius

        u = Vec3List.from_vec3(self.u, len(s))
        v = Vec3List.from_vec3(self.v, len(s))
        offset_list = u.mul_ndarray(rd[0]) + v.mul_ndarray(rd[1])

        origin_list = offset_list + self.origin

        horizontal_multi = Vec3List.from_vec3(self.horizontal, len(s))
        vertical_multi = Vec3List.from_vec3(self.vertical, len(s))
        direction_list = (horizontal_multi.mul_ndarray(s) - vertical_multi.mul_ndarray(t) + self.top_left_corner - self.origin - offset_list)

        return RayList(origin_list, direction_list)
