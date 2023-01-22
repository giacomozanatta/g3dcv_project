from point import *
class Voxel:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class VoxelSet:
    def __init__(self, center, padding, n):
        offset = (padding*2)/n
        self.set = []
        j = 0
        for i in range(0,n+1):
            print(center.x - padding + (i * offset))
            j+=1
            self.set.append(Voxel(center.x - padding + (i * offset), center.x - padding + (i * offset), 0))
        print(j)
VoxelSet(Point(500,500), 300, 100)