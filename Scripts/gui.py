import taichi as ti

ti.init(arch=ti.cuda)


@ti.data_oriented
class Gui:
    def __init__(self, scene):
        # variables
        self.num_rod = scene.num_rod
        self.num_ver = scene.num_ver

        # visulization assistant variables
        self.vertice = ti.Vector.field(
            n=3, dtype=ti.f32, shape=self.num_rod * self.num_ver
        )
        self.indices = ti.field(dtype=int, shape=2 * (self.num_ver - 1))

        # init indices
        for i in range(self.indices.shape[0]):
            self.indices[i] = (i + 1) // 2

        # window and camera
        self.window = ti.ui.Window("Hair DER", (1024, 1024), vsync=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1, 1, 1))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0, -0.25, 5)
        self.camera.lookat(0, -0.25, 0)
        self.scene.set_camera(self.camera)

    @ti.kernel
    def update_vertices(self, X: ti.types.template()):
        for i, j in X:
            self.vertice[i * self.num_ver + j] = X[i, j]

    def setting(self, scene):
        self.scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.scene.ambient_light((0.5, 0.5, 0.5))
        self.scene.particles(self.vertice, radius=scene.r, color=(0, 0, 0))
        self.scene.lines(self.vertice, width=4, indices=self.indices, color=(0, 0, 0))
        self.canvas.scene(self.scene)

        # TODO: multiple strands

    def show(self):
        self.window.show()

    def save_png(self, frame):
        self.window.save_image("output/{}.png".format(frame))

    def window_state(self):
        return self.window.running
