from xmlparse import Scene
from Hair import Hair_DER
import argparse
import taichi as ti

ti.init(arch=ti.cuda)

################################ __main__ ################################

num_ver = 41
num_rod = 1

# visulization
vertice = ti.Vector.field(n=3, dtype=ti.f32, shape=num_rod * num_ver)
indices = ti.field(dtype=int, shape=2 * (num_ver - 1))


@ti.kernel
def update_vertices(X: ti.types.template()):
    for i, j in X:
        vertice[i * num_ver + j] = X[i, j]


@ti.kernel
def init_indices():
    for i in ti.static(range(2 * (num_ver - 1))):
        indices[i] = (i + 1) // 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene", type=str, help="input xml file")
    parser.add_argument("-o", "--outfile", type=str)
    args = parser.parse_args()
    m_Scene = Scene(args.scene)
    m_Scene.initialize_attrib()
    init_indices()
    Hair_Sim = Hair_DER(
        m_Scene.X,
        m_Scene.Theta,
        m_Scene.Fixed,
        m_Scene.num_rod,
        m_Scene.num_ver,
        m_Scene.dt,
        m_Scene.rho,
        m_Scene.r,
        m_Scene.E,
        m_Scene.G,
    )
    Hair_Sim.Init()
    window = ti.ui.Window("Hair DER", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, -0.25, 5)
    camera.lookat(0, -0.25, 0)
    frames = 0
    file = open("Outfiles\outfile.txt", "w")
    while window.running and frames < 1000:
        Hair_Sim.write_to_file(file, frames)
        print("frame:{}".format(frames))
        for _ in range(int(1e-6 // m_Scene.dt)):
            Hair_Sim.step()
        frames += 1
        m_Scene.update(Hair_Sim.X, Hair_Sim.Theta)
        update_vertices(m_Scene.X)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(vertice, radius=m_Scene.r, color=(0, 0, 0))
        scene.lines(
            vertice, width=4, indices=indices, color=(0, 0, 0)
        )  # TODO: multiple strands
        canvas.scene(scene)
        window.save_image("output/{}.png".format(frames))
        window.show()
    file.close()
