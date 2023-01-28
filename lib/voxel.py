from lib.point import *
class Voxel:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class VoxelSet:
    def __init__(self, center: Point3D, padding, n):
        self.offset = (padding*2)/n
        self.set = []
        self.n = n
        for i in range(0,n):
            for j in range(0, n):
                for k in range(0, n):
                    self.set.append([center.x - padding + (i * self.offset), center.y - padding + (j * self.offset), center.z - padding + (k * self.offset)])
