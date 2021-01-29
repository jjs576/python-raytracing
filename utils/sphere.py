import cupy as cp
from utils.vec3 import Vec3, Point3, Vec3List
from utils.ray import RayList
from utils.hittable import Hittable, HitRecordList
from utils.material import Material


class Sphere(Hittable):
    def __init__(self, center, r, mat):
        self.center = center
        self.radius = r
        self.material = mat

    def hit(self, r, t_min, t_max):
        if isinstance(t_max, (int, float, cp.floating)):
            t_max_list = cp.full(len(r), t_max, cp.float32)
        else:
            t_max_list = t_max
        oc = r.origin() - self.center
        a = r.direction().length_squared()
        half_b = oc @ r.direction()
        c = oc.length_squared() - self.radius**2
        discriminant_list = half_b**2 - a*c

        discriminant_condition = discriminant_list > 0
        if not discriminant_condition.any():
            return HitRecordList.new(len(r)).set_compress_info(None)

        positive_discriminant_list = (discriminant_list * discriminant_condition)
        root = cp.sqrt(positive_discriminant_list)
        non_zero_a = a - (a == 0)
        t_0 = (-half_b - root) / non_zero_a
        t_1 = (-half_b + root) / non_zero_a

        t_0_condition = (
            (t_min < t_0) & (t_0 < t_max_list) & discriminant_condition
        )
        t_1_condition = (
            (t_min < t_1) & (t_1 < t_max_list) & (~t_0_condition) & discriminant_condition
        )

        t = cp.where(t_0_condition, t_0, 0)
        t = cp.where(t_1_condition, t_1, t)

        condition = t > 0
        full_rate = condition.sum() / len(t)
        if full_rate > 0.5:
            idx = None
        else:
            idx = cp.where(condition)[0]
            t = t[idx]
            r = RayList(
                Vec3List(r.origin().get_ndarray(idx)),
                Vec3List(r.direction().get_ndarray(idx))
            )

        point = r.at(t)
        outward_normal = (point - self.center) / self.radius
        result = HitRecordList(
            point, t, cp.full(len(r), self.material.idx, dtype=cp.int32)
        ).set_face_normal(r, outward_normal).set_compress_info(idx)
        return result
