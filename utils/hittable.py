import cupy as cp
from abc import ABC, abstractmethod
from utils.vec3 import Vec3, Point3, Vec3List
from utils.ray import Ray, RayList


class HitRecord:
    def __init__(self, point, t, mat_idx, normal = Vec3(), front_face = False):
        self.p = point
        self.t = t
        self.material_idx = mat_idx
        self.normal = normal
        self.front_face = front_face

    def set_face_normal(self, r, outward_normal):
        self.front_face = (r.direction() @ outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
        return self


class HitRecordList:
    def __init__(self, point, t, mat, normal = Vec3List.new_zero(0), front_face = cp.array([])):
        self.p = point
        self.t = t
        self.material = mat
        self.normal = normal
        self.front_face = front_face

    def set_face_normal(self, r, outward_normal):
        self.front_face = (r.direction() * outward_normal).e.sum(axis=1) < 0
        front_face_3 = Vec3List.from_array(self.front_face)
        self.normal = Vec3List(cp.where(front_face_3.e, outward_normal.e, -outward_normal.e))
        return self

    def __getitem__(self, idx):
        mat_idx = self.material[idx]
        if mat_idx == 0:
            return None
        else:
            return HitRecord(self.p[idx], self.t[idx], mat_idx, self.normal[idx], self.front_face[idx])

    def __setitem__(self, idx, rec):
        self.p[idx] = rec.p
        self.t[idx] = rec.t
        self.material[idx] = rec.material_idx
        self.normal[idx] = rec.normal
        self.front_face[idx] = rec.front_face

    def __len__(self):
        return len(self.t)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration
        result = self[self.idx]
        self.idx += 1
        return result

    def set_compress_info(self, idx):
        self.compress_idx = idx
        return self

    def update(self, new):
        if new.compress_idx is not None:
            idx = new.compress_idx
            old_idx = cp.arange(len(idx))
        else:
            idx = slice(0, -1)
            old_idx = slice(0, -1)

        change = (new.t[old_idx] < self.t[idx]) & (new.t[old_idx] > 0)
        if not change.any():
            return self
        change_3 = Vec3List.from_array(change)

        self.p.e[idx] = cp.where(change_3.e, new.p.e[old_idx], self.p.e[idx])
        self.t[idx] = cp.where(change, new.t[old_idx], self.t[idx])
        self.material[idx] = cp.where(change, new.material[old_idx], self.material[idx])
        self.normal.e[idx] = cp.where(change_3.e, new.normal.e[old_idx], self.normal.e[idx])
        self.front_face[idx] = cp.where(change, new.front_face[old_idx], self.front_face[idx])
        return self

    @staticmethod
    def new(length):
        return HitRecordList(
            Vec3List.new_empty(length),
            cp.zeros(length, dtype=cp.float32),
            cp.zeros(length, dtype=cp.int32),
            Vec3List.new_empty(length),
            cp.empty(length, dtype=cp.bool)
        )

    @staticmethod
    def new_from_t(t):
        length = len(t)
        return HitRecordList(
            Vec3List.new_empty(length),
            t,
            cp.zeros(length, dtype=cp.int32),
            Vec3List.new_empty(length),
            cp.empty(length, dtype=cp.bool)
        )

class Hittable(ABC):
    @abstractmethod
    def hit(self, r, t_min, t_max):
        return NotImplemented
