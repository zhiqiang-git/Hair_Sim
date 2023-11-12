from xmlparse import Scene
from Hair import Hair_DER
from gui import Gui
import argparse

################################ __main__ ################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene", type=str, help="input xml file")
    parser.add_argument("-o", "--outfile", type=str)
    args = parser.parse_args()
    m_Scene = Scene(args.scene)
    m_Scene.initialize_attrib()
    gui = Gui(m_Scene)
    Hair_Sim = Hair_DER(m_Scene)
    Hair_Sim.Init()
    frames = 0
    file = open("Outfiles\outfile.txt", "w")
    while gui.window_state() and frames < 10000:
        Hair_Sim.write_to_file(file, frames)
        if(frames == 0):
            Hair_Sim.Interupt_Theta()
            pass
        for _ in range(int(1e-4 // m_Scene.dt)):
            Hair_Sim.Explicit_step(gravity=0)
        frames += 1
        m_Scene.update(Hair_Sim.X, Hair_Sim.Theta)
        gui.update_vertices(m_Scene.X)
        gui.setting(m_Scene)
        gui.show()
        gui.save_png(frames)
    file.close()
