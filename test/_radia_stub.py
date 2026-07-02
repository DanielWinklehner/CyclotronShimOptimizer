"""A minimal in-memory stand-in for the radia module.

Lets the component / field-calculator tests run without a GPU (or radia at
all): object ids are handed out sequentially, containers remember their
members, and Fld() evaluates analytic field callables registered per object id
(containers sum their members' fields). All calls are recorded for assertions.
"""

import numpy as np


class RadiaStub:
    def __init__(self):
        self._next_id = 1000
        self.calls = []       # (funcname, args...) in call order
        self.deleted = []     # ids passed to UtiDel
        self.containers = {}  # container id -> list of member ids
        self.fields = {}      # object id -> callable((N, 3) array) -> (N, 3) array

    # -- object management -------------------------------------------------
    def _new_id(self):
        self._next_id += 1
        return self._next_id

    def ObjCnt(self, ids):
        cid = self._new_id()
        self.containers[cid] = list(ids)
        self.calls.append(("ObjCnt", tuple(ids), cid))
        return cid

    def UtiDel(self, oid):
        self.deleted.append(oid)
        self.containers.pop(oid, None)
        self.calls.append(("UtiDel", oid))
        return 0

    def UtiDelAll(self):
        self.calls.append(("UtiDelAll",))
        return 0

    # -- transforms / attributes (recorded only) ----------------------------
    def TrfZerPerp(self, oid, point, normal):
        self.calls.append(("TrfZerPerp", oid, tuple(point), tuple(normal)))
        return oid

    def TrfZerPara(self, oid, point, normal):
        self.calls.append(("TrfZerPara", oid, tuple(point), tuple(normal)))
        return oid

    def MatApl(self, oid, material):
        self.calls.append(("MatApl", oid, material))
        return oid

    def ObjDrwAtr(self, oid, color):
        self.calls.append(("ObjDrwAtr", oid, tuple(color)))
        return 0

    # -- field evaluation ----------------------------------------------------
    def register_field(self, oid, func):
        """Attach an analytic field callable((N,3)) -> (N,3) to an object id."""
        self.fields[oid] = func

    def _field_of(self, oid):
        if oid in self.fields:
            return self.fields[oid]
        if oid in self.containers:
            members = list(self.containers[oid])

            def summed(pts, _members=members):
                total = np.zeros((len(pts), 3))
                for member in _members:
                    total += self._field_of(member)(pts)
                return total

            return summed
        raise KeyError(f"No field registered for radia stub id {oid}")

    def Fld(self, oid, components, points, use_gpu=True, precision="double"):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        b = np.asarray(self._field_of(oid)(pts), dtype=float)
        self.calls.append(("Fld", oid, components, len(pts), precision))
        if components == "bz":
            return b[:, 2].tolist()
        if components == "b":
            return b.tolist()
        # NOTE: no 'bxbybz' -- RadiaCUDA's GPU gate only accepts b/bx/by/bz,
        # so production code must use 'b'; keep the stub equally strict.
        raise ValueError(f"RadiaStub.Fld: unsupported components {components!r}")
