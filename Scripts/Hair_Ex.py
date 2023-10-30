import taichi as ti
import numpy as np
import sys

ti.init(arch=ti.cuda)


################################### Hair_DER ###############################
@ti.data_oriented
class Hair_DER:
    def __init__(self, scene):
        # variables used in this class
        # basic parameters
        self.num_rod = scene.num_rod
        self.num = scene.num_ver
        self.dt = scene.dt
        self.rho = scene.rho
        self.r = scene.r
        self.E = scene.E
        self.G = scene.G
        self.damping = scene.damping

        # vertices parameters
        self.X = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num))
        self.V = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num))
        self.F = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num))
        self.Fixed = ti.field(dtype=ti.i32, shape=(self.num_rod, self.num))
        self.m = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num))

        # edges(angels) parameters
        self.Theta = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.Omega = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.Tao = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.I = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))

        # framework(reference and material)
        self.a1 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.a2 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.m1 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.m2 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))

        # variables initial_version
        self.i_edge_length = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.i_tangent = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1)
        )
        self.i_ver_length = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num))
        self.i_kb = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 2)
        )
        self.i_k = ti.Vector.field(
            n=4, dtype=ti.f32, shape=(self.num_rod, self.num - 2)
        )
        self.i_miu = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))

        # variables
        self.edge_length = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.tangent = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1)
        )
        self.tangent_old = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1)
        )  # for temporal_trans
        self.ver_length = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num))
        self.kb = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 2))
        self.k = ti.Vector.field(n=4, dtype=ti.f32, shape=(self.num_rod, self.num - 2))
        self.ref_twist = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.miu = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))

        # grad field
        self.grad_bend = ti.Matrix.field(
            n=4, m=3, dtype=ti.f32, shape=(self.num_rod, self.num - 2, 2)
        )
        self.grad_bend_tor = ti.Vector.field(
            n=4, dtype=ti.f32, shape=(self.num_rod, self.num - 2, 2)
        )
        self.grad_twist = ti.Vector.field(
            n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 2, 2)
        )

        ###############################################
        # basic initialization
        for i in range(self.num_rod):
            for j in range(self.num):
                self.F[i, j] = ti.Vector([0, 0, 0])
                self.V[i, j] = ti.Vector([0, 0, 0])
                self.X[i, j] = scene.X[i, j]
                self.Fixed[i, j] = scene.Fixed[i, j]
                if j != self.num - 1:
                    self.Tao[i, j] = 0
                    self.Omega[i, j] = 0
                    self.Theta[i, j] = scene.Theta[i, j]

    ####################################################
    # some private methods
    @ti.kernel
    def Clear(self):
        for i, j in self.X:
            self.F[i, j] = ti.Vector([0, 0, 0])
            if j != self.num - 1:
                self.Tao[i, j] = 0

    @ti.func
    def Trans_frame(
        self, a: ti.types.template(), t0: ti.types.template(), t1: ti.types.template()
    ) -> ti.types.template():
        b = ti.math.cross(t0, t1)
        ret = ti.Vector([0.0, 0.0, 0.0])
        if b.norm() < sys.float_info.epsilon:
            ret = a
        else:
            b = ti.math.normalize(b)

            n0 = ti.math.cross(t0, b)
            n1 = ti.math.cross(t1, b)
            n0 = ti.math.normalize(n0)
            n1 = ti.math.normalize(n1)

            i = ti.math.dot(a, b)
            j = ti.math.dot(a, n0)

            ret = i * b + j * n1
        return ret

    @ti.func
    def Rotate(
        self, t: ti.types.template(), v: ti.types.template(), angel: ti.f32
    ) -> ti.types.template():
        b = ti.math.cross(t, v)

        return ti.math.cos(angel) * v + ti.math.sin(angel) * b

    @ti.func
    def Angel(
        self, v0: ti.types.template(), v1: ti.types.template(), t: ti.types.template()
    ) -> ti.f32:
        n0 = ti.math.normalize(v0)
        n1 = ti.math.normalize(v1)

        b0 = ti.math.cross(t, n0)
        b0 = ti.math.normalize(b0)

        a = ti.math.dot(n1, n0)
        b = ti.math.dot(n1, b0)

        angel = 0.0

        if a >= 1:
            angel = 0
        elif a <= -1:
            angel = 180 / np.pi
        else:
            angel = ti.math.acos(a)

        if b < 0:
            angel = -angel

        return angel

    ####################################################
    # initialization
    @ti.kernel
    def Init(self):
        for _ in range(1):
            for i in range(self.num_rod):
                for j in range(self.num - 1):
                    self.i_tangent[i, j] = self.X[i, j + 1] - self.X[i, j]
                    self.i_tangent[i, j] = ti.math.normalize(self.i_tangent[i, j])
                    self.i_edge_length[i, j] = (self.X[i, j + 1] - self.X[i, j]).norm()
                    self.edge_length[i, j] = self.i_edge_length[i, j]
                    self.tangent[i, j] = self.i_tangent[i, j]
                    self.tangent_old[i, j] = self.i_tangent[i, j]
                    self.I[i, j] = (
                        0.5
                        * self.edge_length[i, j]
                        * (np.pi)
                        * (self.r**4)
                        * self.rho
                    )

            for i in range(self.num_rod):
                for j in range(self.num - 1):
                    # frame initialization
                    if j == 0:
                        rand_vec_1 = ti.Vector([1.0, 0, 0])  # a normalized vector
                        rand_vec_2 = ti.Vector([0, 1.0, 0])
                        if (
                            ti.math.cross(self.i_tangent[i, j], rand_vec_1).norm()
                            < sys.float_info.epsilon
                        ):
                            self.a1[i, j] = ti.math.cross(
                                self.i_tangent[i, j], rand_vec_2
                            )
                        else:
                            self.a1[i, j] = ti.math.cross(
                                self.i_tangent[i, j], rand_vec_1
                            )
                    else:
                        self.a1[i, j] = self.Trans_frame(
                            self.a1[i, j - 1],
                            self.i_tangent[i, j - 1],
                            self.i_tangent[i, j],
                        )
                    self.a1[i, j] = ti.math.normalize(self.a1[i, j])
                    self.a2[i, j] = ti.math.cross(self.i_tangent[i, j], self.a1[i, j])
                    self.a2[i, j] = ti.math.normalize(self.a2[i, j])
                    self.m1[i, j] = self.a1[i, j] * ti.math.cos(
                        self.Theta[i, j]
                    ) + self.a2[i, j] * ti.math.sin(self.Theta[i, j])
                    self.m2[i, j] = self.a1[i, j] * (
                        -ti.math.sin(self.Theta[i, j])
                    ) + self.a2[i, j] * ti.math.cos(self.Theta[i, j])

            for i in range(self.num_rod):
                for j in range(self.num):
                    # ver_length
                    if j != 0 and j != self.num - 1:
                        self.i_ver_length[i, j] = (
                            self.i_edge_length[i, j - 1] + self.i_edge_length[i, j]
                        ) / 2
                    elif j == 0:
                        self.i_ver_length[i, j] = self.i_edge_length[i, j] / 2
                    elif j == self.num - 1:
                        self.i_ver_length[i, j] = self.i_edge_length[i, j - 1] / 2
                    self.ver_length[i, j] = self.i_ver_length[i, j]

                    # ver_mass
                    if j != 0 and j != self.num - 1:
                        self.m[i, j] = (
                            0.5
                            * self.rho
                            * np.pi
                            * (self.r**2)
                            * (self.edge_length[i, j - 1] + self.edge_length[i, j])
                        )
                    elif j == 0:
                        self.m[i, j] = (
                            0.5
                            * self.rho
                            * np.pi
                            * (self.r**2)
                            * self.edge_length[i, j]
                        )
                    elif j == self.num - 1:
                        self.m[i, j] = (
                            0.5
                            * self.rho
                            * np.pi
                            * (self.r**2)
                            * self.edge_length[i, j - 1]
                        )

            for i in range(self.num_rod):
                for j in range(self.num - 2):
                    # bend force variables
                    self.i_kb[i, j] = (
                        2
                        * (
                            ti.math.cross(
                                self.i_tangent[i, j], self.i_tangent[i, j + 1]
                            )
                        )
                        / (
                            1
                            + ti.math.dot(
                                self.i_tangent[i, j], self.i_tangent[i, j + 1]
                            )
                        )
                    )
                    self.i_k[i, j] = ti.Vector(
                        [
                            ti.math.dot(self.m2[i, j], self.i_kb[i, j]),
                            ti.math.dot(self.m2[i, j + 1], self.i_kb[i, j]),
                            -ti.math.dot(self.m1[i, j], self.i_kb[i, j]),
                            -ti.math.dot(self.m1[i, j + 1], self.i_kb[i, j]),
                        ]
                    )

            for i in range(self.num_rod):
                for j in range(self.num - 1):
                    # twist force variables
                    if j == 0:
                        self.i_miu[i, j] = 0  # bounding condition: Theta[0] = Theta[-1]
                    else:
                        self.i_miu[i, j] = self.Theta[i, j] - self.Theta[i, j - 1]
                    self.ref_twist[i, j] = 0
                    self.miu[i, j] = self.i_miu[i, j]

    #################################################################
    # update variables
    @ti.kernel
    def Update_tan(self):
        for i, j in self.Theta:
            self.tangent_old[i, j] = self.tangent[i, j]
            self.tangent[i, j] = self.X[i, j + 1] - self.X[i, j]
            self.tangent[i, j] = ti.math.normalize(self.tangent[i, j])
            self.edge_length[i, j] = (self.X[i, j + 1] - self.X[i, j]).norm()

    @ti.kernel
    def Update_frame(self):
        for i, j in self.Theta:
            self.a1[i, j] = self.Trans_frame(
                self.a1[i, j], self.tangent_old[i, j], self.tangent[i, j]
            )
            self.a1[i, j] = ti.math.normalize(self.a1[i, j])
            self.a2[i, j] = ti.math.cross(self.tangent[i, j], self.a1[i, j])
            self.a2[i, j] = ti.math.normalize(self.a2[i, j])
            self.m1[i, j] = self.a1[i, j] * ti.math.cos(self.Theta[i, j]) + self.a2[
                i, j
            ] * ti.math.sin(self.Theta[i, j])
            self.m2[i, j] = self.a1[i, j] * (-ti.math.sin(self.Theta[i, j])) + self.a2[
                i, j
            ] * ti.math.cos(self.Theta[i, j])

    @ti.kernel
    def Update_ver_l(self):
        for i, j in self.X:
            # ver_length
            if j != 0 and j != self.num - 1:
                self.ver_length[i, j] = (
                    self.edge_length[i, j - 1] + self.edge_length[i, j]
                ) / 2
            elif j == 0:
                self.ver_length[i, j] = self.edge_length[i, j] / 2
            elif j == self.num - 1:
                self.ver_length[i, j] = self.edge_length[i, j - 1] / 2

    @ti.kernel
    def Update_bend(self):
        for i, j in self.kb:
            self.kb[i, j] = (
                2
                * (ti.math.cross(self.tangent[i, j], self.tangent[i, j + 1]))
                / (1 + ti.math.dot(self.tangent[i, j], self.tangent[i, j + 1]))
            )
            self.k[i, j] = ti.Vector(
                [
                    ti.math.dot(self.m2[i, j], self.kb[i, j]),
                    ti.math.dot(self.m2[i, j + 1], self.kb[i, j]),
                    -ti.math.dot(self.m1[i, j], self.kb[i, j]),
                    -ti.math.dot(self.m1[i, j + 1], self.kb[i, j]),
                ]
            )

    @ti.kernel
    def Update_twist(self):
        for i, j in self.Theta:
            if j == 0:
                self.ref_twist[i, j] = 0
                self.miu[i, j] = 0  # bounding condition
            else:
                a_space = self.Trans_frame(
                    self.a1[i, j - 1], self.tangent[i, j - 1], self.tangent[i, j]
                )
                a_space = self.Rotate(self.tangent[i, j], a_space, self.ref_twist[i, j])

                angel = self.Angel(a_space, self.a1[i, j], self.tangent[i, j])

                self.ref_twist[i, j] += angel

                self.miu[i, j] = self.ref_twist[i, j] + (
                    self.Theta[i, j] - self.Theta[i, j - 1]
                )

    ########################################################################
    # calculate gradients and forces(for vertices)
    @ti.kernel
    def Gravity(self, G: ti.types.template()):
        for i, j in self.X:
            self.F[i, j] += self.m[i, j] * G

    @ti.kernel
    def Stretch(self):
        for i, j in self.X:
            if j != 0:
                self.F[i, j] -= (
                    np.pi
                    * (self.r**2)
                    * self.E
                    * (self.edge_length[i, j - 1] / self.i_edge_length[i, j - 1] - 1)
                    * self.tangent[i, j - 1]
                )
            if j != self.num - 1:
                self.F[i, j] += (
                    np.pi
                    * (self.r**2)
                    * self.E
                    * (self.edge_length[i, j] / self.i_edge_length[i, j] - 1)
                    * self.tangent[i, j]
                )

    @ti.kernel
    def Bend_Gradient(self):
        for i, j in self.kb:
            bot = 1 + ti.math.dot(self.tangent[i, j], self.tangent[i, j + 1])
            t_hat = (self.tangent[i, j] + self.tangent[i, j + 1]) / bot
            el = self.edge_length[i, j]
            er = self.edge_length[i, j + 1]
            # row1
            row_l1 = (
                -self.k[i, j][0] * t_hat
                + 2 * ti.math.cross(self.tangent[i, j + 1], self.m2[i, j]) / bot
            ) / el
            row_r1 = (
                -self.k[i, j][0] * t_hat
                - 2 * ti.math.cross(self.tangent[i, j], self.m2[i, j]) / bot
            ) / er
            # row2
            row_l2 = (
                -self.k[i, j][1] * t_hat
                + 2 * ti.math.cross(self.tangent[i, j + 1], self.m2[i, j + 1]) / bot
            ) / el
            row_r2 = (
                -self.k[i, j][1] * t_hat
                - 2 * ti.math.cross(self.tangent[i, j], self.m2[i, j + 1]) / bot
            ) / er
            # row3
            row_l3 = (
                -self.k[i, j][2] * t_hat
                - 2 * ti.math.cross(self.tangent[i, j + 1], self.m1[i, j]) / bot
            ) / el
            row_r3 = (
                -self.k[i, j][2] * t_hat
                + 2 * ti.math.cross(self.tangent[i, j], self.m1[i, j]) / bot
            ) / er
            # row4
            row_l4 = (
                -self.k[i, j][3] * t_hat
                - 2 * ti.math.cross(self.tangent[i, j + 1], self.m1[i, j + 1]) / bot
            ) / el
            row_r4 = (
                -self.k[i, j][3] * t_hat
                + 2 * ti.math.cross(self.tangent[i, j], self.m1[i, j + 1]) / bot
            ) / er

            self.grad_bend[i, j, 0] = ti.Matrix.rows([row_l1, row_l2, row_l3, row_l4])
            self.grad_bend[i, j, 1] = ti.Matrix.rows([row_r1, row_r2, row_r3, row_r4])

    @ti.kernel
    def Bend_Force(self):
        for i, j in self.kb:
            cons = self.E * np.pi * (self.r**4) / (8 * self.ver_length[i, j + 1])

            Fl = cons * (
                self.grad_bend[i, j, 0].transpose() @ (self.k[i, j] - self.i_k[i, j])
            )
            Fr = cons * (
                self.grad_bend[i, j, 1].transpose() @ (self.k[i, j] - self.i_k[i, j])
            )

            self.F[i, j] -= -Fl
            self.F[i, j + 1] -= Fl - Fr
            self.F[i, j + 2] -= Fr

    @ti.kernel
    def Twist_Gradient(self):
        for i, j in self.kb:
            self.grad_twist[i, j, 0] = self.kb[i, j] / (2 * self.edge_length[i, j])
            self.grad_twist[i, j, 1] = self.kb[i, j] / (2 * self.edge_length[i, j + 1])

    @ti.kernel
    def Twist_Force(self):
        for i, j in self.kb:
            cons = self.G * np.pi * (self.r**4) / (4 * self.ver_length[i, j + 1])
            Fl = (
                cons
                * (self.miu[i, j + 1] - self.i_miu[i, j + 1])
                * self.grad_twist[i, j, 0]
            )
            Fr = (
                cons
                * (self.miu[i, j + 1] - self.i_miu[i, j + 1])
                * self.grad_twist[i, j, 1]
            )

            self.F[i, j] -= -Fl
            self.F[i, j + 1] -= Fl - Fr
            self.F[i, j + 2] -= Fr

    #######################################################
    # calculate gradients and torques(for angels)
    @ti.kernel
    def Bend_Grad_Torque(self):
        for i, j in self.kb:
            self.grad_bend_tor[i, j, 0] = ti.Vector(
                [
                    -ti.math.dot(self.m1[i, j], self.kb[i, j]),
                    0,
                    -ti.math.dot(self.m2[i, j], self.kb[i, j]),
                    0,
                ]
            )
            self.grad_bend_tor[i, j, 1] = ti.Vector(
                [
                    0,
                    -ti.math.dot(self.m1[i, j + 1], self.kb[i, j]),
                    0,
                    -ti.math.dot(self.m2[i, j + 1], self.kb[i, j]),
                ]
            )

    @ti.kernel
    def Bend_Torque(self):
        for i, j in self.kb:
            cons = self.E * np.pi * (self.r**4) / (8 * self.ver_length[i, j + 1])
            Tl = cons * ti.math.dot(
                self.grad_bend_tor[i, j, 0], self.k[i, j] - self.i_k[i, j]
            )
            Tr = cons * ti.math.dot(
                self.grad_bend_tor[i, j, 1], self.k[i, j] - self.i_k[i, j]
            )

            self.Tao[i, j] -= Tl
            self.Tao[i, j + 1] -= Tr

    @ti.kernel
    def Twist_Torque(self):
        for i, j in self.kb:
            cons = self.G * np.pi * (self.r**4) / (4 * self.ver_length[i, j + 1])
            Tl = cons * (-1) * (self.miu[i, j + 1] - self.i_miu[i, j + 1])
            Tr = cons * (1) * (self.miu[i, j + 1] - self.i_miu[i, j + 1])

            self.Tao[i, j] -= Tl
            self.Tao[i, j + 1] -= Tr

    ########################################################
    # integrator
    @ti.kernel
    def Explicit_Ver(self):
        for i, j in self.X:
            if not self.Fixed[i, j]:
                self.V[i, j] += self.dt * (self.F[i, j] / self.m[i, j])
                # self.V[i, j] *= self.damping
                self.X[i, j] += self.dt * self.V[i, j]

    @ti.kernel
    def Explicit_Angel(self):
        for i, j in self.Theta:
            self.Omega[i, j] += self.dt * (self.Tao[i, j] / self.I[i, j])
            # self.Omega[i, j] *= self.damping
            self.Theta[i, j] += self.dt * self.Omega[i, j]

    def step(self):
        # set forces and torques to zero
        self.Clear()

        # calculate forces
        # self.Gravity(ti.Vector([0.0, -981, 0.0]))
        self.Stretch()
        self.Bend_Gradient()
        self.Bend_Force()
        self.Twist_Gradient()
        self.Twist_Force()

        # calculate torques
        self.Bend_Grad_Torque()
        self.Bend_Torque()
        self.Twist_Torque()

        # explicit integration
        self.Explicit_Ver()
        self.Explicit_Angel()

        # update variables
        self.Update_tan()
        self.Update_ver_l()
        self.Update_frame()
        self.Update_bend()
        self.Update_twist()
        # print("Done")

    def write_to_file(self, outfile, frame):
        outfile.write("------frame {}-----\n".format(frame))
        outfile.write("position:\n{}\n".format(self.X))
        # outfile.write("velocity:\n{}\n".format(self.V))
        outfile.write("omega:\n{}\n".format(self.Omega))
        outfile.write("theta:\n{}\n".format(self.Theta))
        outfile.write("ref_twist:\n{}\n".format(self.ref_twist))
        # outfile.write("mass:\n{}\n".format(self.m))
        # outfile.write("edge length:\n{}\n".format(self.edge_length))
        outfile.write("miu:\n{}\n".format(self.miu))
        # outfile.write('streching force:\n{}\n'.format(f_strech))
        # outfile.write('bending force:\n{}\n'.format(f_bend))
        # outfile.write('twisting force:\n{}\n'.format(f_twist))
