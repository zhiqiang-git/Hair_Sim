import taichi as ti
import xml.etree.ElementTree as ET

ti.init(arch=ti.cuda)


@ti.data_oriented
class Scene:
    def __init__(self, path):
        self.dt = 1e-7
        self.tree = ET.parse(path)
        root = self.tree.getroot()
        i = 0
        for child in root:
            if child.tag == "StrandParameters":
                self.E = float(child.find("youngsModulus").attrib["value"])
                self.G = float(child.find("shearModulus").attrib["value"])
                self.rho = float(child.find("density").attrib["value"])
                self.r = float(child.find("radius").attrib["value"])
            if child.tag == "Strand":
                if i == 0:
                    self.num_ver = child.__len__()
                i += 1
        self.num_rod = i

        self.X = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num_ver))
        self.Fixed = ti.field(dtype=ti.i32, shape=(self.num_rod, self.num_ver))
        self.Theta = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num_ver - 1))

    def initialize_strand(self, strand, i):
        j = 0
        for p in strand:
            assert p.tag == "particle"
            self.X[i, j] = ti.Vector([float(x) for x in p.attrib["x"].split()])
            if p.attrib["fixed"] == "1":
                self.Fixed[i, j] = 1
            else:
                self.Fixed[i, j] = 0
            j += 1

    def initialize_theta(self):
        for i in range(self.num_rod):
            for j in range(self.num_ver - 1):
                self.Theta[i, j] = 0

    def initialize_attrib(self):
        root = self.tree.getroot()
        i = 0
        for child in root:
            if child.tag == "Strand":
                self.initialize_strand(child, i)
                i += 1
        self.initialize_theta()

    def update(self, X, Theta):
        self.X = X
        self.Theta = Theta
