import taichi as ti
import taichi.math as tm
import sys

ti.init(arch=ti.cuda)

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
        self.Gravity = scene.Gravity

        # vertices parameters
        self.X = scene.X
        self.V = scene.V
        self.f_stretch = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num))
        self.f_bend = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num))
        self.f_twist = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num))
        self.f_gravity = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num))
        self.Fixed = scene.Fixed
        self.m = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num))

        # edges(angels) parameters
        self.Theta = scene.Theta
        self.Omega = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.tau_bend = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.tau_twist = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.I = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))

        # framework(reference and material)
        self.a1 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.a2 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.m1 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.m2 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))

        # variables initial_version
        # rest shape means stable, do not change over time
        # we need extra functions to add some pulse
        self.rest_edge_length = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.rest_tangent = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.rest_ver_length = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num))
        self.rest_curvature_binormal = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 2))
        self.rest_kappa = ti.Vector.field(n=2, dtype=ti.f32, shape=(self.num_rod, self.num - 2))
        self.rest_twist = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 2))

        # variables
        self.edge_length = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.tangent = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))
        self.tangent_old = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 1))  # for temporal_trans
        self.ver_length = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num))
        self.curvature_binormal = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.num_rod, self.num - 2))
        self.kappa = ti.Vector.field(n=2, dtype=ti.f32, shape=(self.num_rod, self.num - 2))                    # bend curvature
        self.ref_twist = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 2))   # angle between a_k+1(spacial para)(bishop) and a_k+1(temporal para)
        self.twist = ti.field(dtype=ti.f32, shape=(self.num_rod, self.num - 2))       # angle between mat frame and bishop frame

        # grad of kappa and twist
        self.grad_kappa = ti.Matrix.field(n=11, m=2, dtype=ti.f32, shape=(self.num_rod, self.num - 2))
        self.grad_twist = ti.Vector.field(n=11, dtype=ti.f32, shape=(self.num_rod, self.num - 2))

    @ti.kernel
    def Clear(self):
        for i, j in self.X:
            self.f_bend[i, j] = ti.Vector([0, 0, 0])
            self.f_stretch[i, j] = ti.Vector([0, 0, 0])
            self.f_twist[i, j] = ti.Vector([0, 0, 0])
            self.f_gravity[i, j] = ti.Vector([0, 0, 0])
            if j != self.num - 1:
                self.tau_bend[i, j] = 0
                self.tau_twist[i, j] = 0

    @ti.func
    def Trans_frame(self, a: ti.types.template(), t0: ti.types.template(), t1: ti.types.template()) -> ti.types.template():
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
        return tm.normalize(ret)
    
    @ti.func
    def Rotate(self, t: ti.types.template(), v: ti.types.template(), angel: ti.f32) -> ti.types.template():
        b = tm.cross(t, v)
        b = tm.normalize(b)
        return tm.cos(angel) * v + tm.sin(angel) * b

    @ti.func
    def Angle(self, v0: ti.types.template(), v1: ti.types.template(), t: ti.types.template()) -> ti.f32:
        n0 = tm.normalize(v0)
        n1 = tm.normalize(v1)
        b0 = tm.cross(t, n0)
        n0 = tm.normalize(n0)
        a = tm.dot(n1, n0)
        b = tm.dot(n1, b0)
        angel = 0.0
        if a >= 1:
            angel = 0
        elif a <= -1:
            angel = 180 / tm.pi
        else:
            angel = tm.acos(a)
        if b < 0:
            angel = -angel
        return angel

    @ti.kernel
    def Init(self):
        for _ in range(1):
            for i in range(self.num_rod):
                for j in range(self.num - 1):
                    self.rest_tangent[i, j] = tm.normalize(self.X[i, j+1] - self.X[i, j])
                    self.tangent[i, j] = self.rest_tangent[i, j]
                    self.rest_edge_length[i, j] = tm.length(self.X[i, j+1] - self.X[i, j])
                    self.I[i, j] = (0.5* self.rest_edge_length[i, j]* (tm.pi)* (self.r**4)* self.rho)
                    self.rest_ver_length[i, j] += 0.5 * self.rest_edge_length[i, j]
                    self.rest_ver_length[i, j+1] += 0.5 * self.rest_edge_length[i, j]

            for i in range(self.num_rod):
                for j in range(self.num - 1):
                    # frame initialization
                    tf = self.rest_tangent[i, j]
                    if j == 0:
                        rand_vec_1 = ti.Vector([1.0, 0, 0])  # a normalized vector
                        rand_vec_2 = ti.Vector([0, 1.0, 0])
                        if (tm.cross(tf, rand_vec_1).norm() < sys.float_info.epsilon):
                            self.a1[i, j] = tm.cross(tf, rand_vec_2)
                        else:
                            self.a1[i, j] = tm.cross(tf, rand_vec_1)
                    else:
                        te = self.rest_tangent[i, j-1]
                        self.a1[i, j] = self.Trans_frame(self.a1[i, j-1], te, tf)
                    theta = self.Theta[i, j]
                    self.a2[i, j] = tm.cross(tf, self.a1[i, j])
                    self.m1[i, j] = self.a1[i, j] * tm.cos(theta) + self.a2[i, j] * tm.sin(theta)
                    self.m2[i, j] = self.a1[i, j] * (-tm.sin(theta)) + self.a2[i, j] * tm.cos(theta)

            for i in range(self.num_rod):
                for j in range(self.num):
                    # ver_mass
                    self.m[i, j] = (self.rho * tm.pi * (self.r**2) * (self.rest_ver_length[i, j]))
                    

            for i in range(self.num_rod):
                for j in range(self.num - 2):
                    # bend force variables
                    te = self.rest_tangent[i, j]
                    tf = self.rest_tangent[i, j+1]
                    m1e = self.m1[i, j]
                    m1f = self.m1[i, j+1]
                    m2e = self.m2[i, j]
                    m2f = self.m2[i, j+1]
                    self.rest_curvature_binormal[i, j] = 2 * tm.cross(te, tf) / (1 + tm.dot(te, tf))
                    kb = self.rest_curvature_binormal[i, j]
                    self.rest_kappa[i, j] = ti.Vector([0.5 * tm.dot(m2e + m2f, kb), -0.5 * tm.dot(m1e + m1f, kb)])

            for i in range(self.num_rod):
                for j in range(self.num - 2):
                    # twist force variables
                    # omit twist0 because it is always 0
                    self.rest_twist[i, j] = self.Theta[i, j+1] - self.Theta[i, j]
                    self.ref_twist[i, j] = 0

    @ti.kernel
    def Update_tangent_related(self):
        for i, j in self.tangent:
            self.tangent_old[i, j] = self.tangent[i, j]
            self.tangent[i, j] = tm.normalize(self.X[i, j+1] - self.X[i, j])
            self.edge_length[i, j] = tm.length(self.X[i, j+1] - self.X[i, j])
            self.ver_length[i, j] += 0.5 * self.edge_length[i, j]
            self.ver_length[i, j+1] += 0.5 * self.edge_length[i, j]

    @ti.kernel
    def Update_frame(self):
        for i, j in self.a1:
            theta = self.Theta[i, j]
            self.a1[i, j] = self.Trans_frame(self.a1[i, j], self.tangent_old[i, j], self.tangent[i, j])
            self.a1[i, j] = tm.normalize(self.a1[i, j])
            self.a2[i, j] = self.Trans_frame(self.a2[i, j], self.tangent_old[i, j], self.tangent[i, j])
            self.a2[i, j] = tm.normalize(self.a2[i, j])
            self.m1[i, j] = self.a1[i, j] * tm.cos(theta) + self.a2[i, j] * tm.sin(theta)
            self.m2[i, j] = self.a1[i, j] * (-tm.sin(theta)) + self.a2[i, j] * tm.cos(theta)
    
    @ti.kernel
    def Update_curvature(self):
        for i, j in self.curvature_binormal:
            te = self.tangent[i, j]
            tf = self.tangent[i, j+1]
            m1e = self.m1[i, j]
            m1f = self.m1[i, j+1]
            m2e = self.m2[i, j]
            m2f = self.m2[i, j+1]
            self.curvature_binormal[i, j] = 2 * tm.cross(te, tf) / (1 + tm.dot(te, tf))
            kb = self.curvature_binormal[i, j]
            self.kappa[i, j] = ti.Vector([0.5 * tm.dot(m2e + m2f, kb), -0.5 * tm.dot(m1e + m1f, kb)])
        
    @ti.kernel
    def Update_twist(self):
        for i, j in self.twist:
            a_space = self.Trans_frame(self.a1[i, j], self.tangent[i, j], self.tangent[i, j+1])
            a_space = self.Rotate(self.tangent[i, j+1], a_space, self.ref_twist[i, j])
            angle = self.Angle(a_space, self.a1[i, j+1], self.tangent[i, j+1])
            self.ref_twist[i, j] += angle
            self.twist[i, j] = self.ref_twist[i, j] + (self.Theta[i, j+1] - self.Theta[i, j])

    @ti.kernel
    def Compute_Stretch(self):
        for i, j in self.edge_length:
            k = tm.pi * (self.r ** 2) * self.E
            l = self.edge_length[i, j]
            l_bar = self.rest_edge_length[i, j]
            f = k * (l / l_bar - 1) * self.tangent[i, j]
            self.f_stretch[i, j] += f
            self.f_stretch[i, j+1] -= f
    
    @ti.kernel
    def Compute_DkappaDx(self):
        for i, j in self.kappa:
            te = self.tangent[i, j]
            tf = self.tangent[i, j+1]
            le = self.edge_length[i, j]
            lf = self.edge_length[i, j+1]
            m1e = self.m1[i, j]
            m1f = self.m1[i, j+1]
            m2e = self.m2[i, j]
            m2f = self.m2[i, j+1]
            kappa = self.kappa[i, j]
            kb = self.curvature_binormal[i, j]
            chi = 1.0 + tm.dot(te, tf)
            if chi <= 0:
                chi = 1e-12
            tilde_t = (te + tf) / chi

            # le means left edge, lf means right edge    (l=edge)
            # te means left theta, tf means right theta  (t=theta)
            Dkappa0Dle = (-kappa[0] * tilde_t + tm.cross(tf, (m2e + m2f) / chi)) / le
            Dkappa0Dlf = (-kappa[0] * tilde_t - tm.cross(te, (m2e + m2f) / chi)) / lf
            Dkappa1Dle = (-kappa[1] * tilde_t - tm.cross(tf, (m1e + m1f) / chi)) / le
            Dkappa1Dlf = (-kappa[1] * tilde_t + tm.cross(te, (m1e + m1f) / chi)) / lf
            Dkappa0Dte = -0.5 * tm.dot(m1e, kb)
            Dkappa0Dtf = -0.5 * tm.dot(m1f, kb)
            Dkappa1Dte = -0.5 * tm.dot(m2e, kb)
            Dkappa1Dtf = -0.5 * tm.dot(m2f, kb)

            # DkappaDx stores DkappaDx = DkappaDl * DlDx and DkappaDt 
            # it should be 4n-1 * 2, but only 11 * 2 part is not zero
            for idx in ti.static(range(3)):
                self.grad_kappa[i, j][idx, 0] = -Dkappa0Dle[idx]
                self.grad_kappa[i, j][4+idx, 0] = Dkappa0Dle[idx] - Dkappa0Dlf[idx]
                self.grad_kappa[i, j][8+idx, 0] = Dkappa0Dlf[idx]
                self.grad_kappa[i, j][idx, 1] = -Dkappa1Dle[idx]
                self.grad_kappa[i, j][4+idx, 1] = Dkappa1Dle[idx] - Dkappa1Dlf[idx]
                self.grad_kappa[i, j][8+idx, 1] = Dkappa1Dlf[idx]
            self.grad_kappa[i, j][3, 0] = Dkappa0Dte
            self.grad_kappa[i, j][7, 0] = Dkappa0Dtf
            self.grad_kappa[i, j][3, 1] = Dkappa1Dte
            self.grad_kappa[i, j][7, 1] = Dkappa1Dtf
            
    @ti.kernel
    def Compute_Bend_Gradient(self):
        for i, j in self.kappa:
            B = self.E * tm.pi * (self.r ** 4) / (8 * self.rest_ver_length[i, j+1])
            f = -B * (self.grad_kappa[i, j] @ (self.kappa[i, j] - self.rest_kappa[i, j]))
            self.f_bend[i, j] += ti.Vector([f[0], f[1], f[2]])
            self.f_bend[i, j+1] += ti.Vector([f[4], f[5], f[6]])
            self.f_bend[i, j+2] += ti.Vector([f[8], f[9], f[10]])
            self.tau_bend[i, j] += f[3]
            self.tau_bend[i, j+1] += f[7]

    @ti.kernel
    def Compute_DtwistDx(self):
        for i, j in self.twist:
            le = self.edge_length[i, j]
            lf = self.edge_length[i, j+1]
            kb = self.curvature_binormal[i, j]
            DtwistDle = kb / (2 * le)
            DtwistDlf = kb / (2 * lf)
            DtwistDte = -1
            DtwistDtf = 1
            for idx in ti.static(range(3)):
                self.grad_twist[i, j][idx] = -DtwistDle[idx]
                self.grad_twist[i, j][4+idx] = DtwistDle[idx] - DtwistDlf[idx]
                self.grad_twist[i, j][8+idx] = DtwistDlf[idx]
            self.grad_twist[i, j][3] = DtwistDte
            self.grad_twist[i, j][7] = DtwistDtf

    @ti.kernel
    def Compute_Twist_Gradient(self):
        for i, j in self.twist:
            C = self.G * tm.pi * (self.r ** 4) / (4 * self.rest_ver_length[i, j+1])
            f = -C * (self.grad_twist[i, j]) * (self.twist[i, j] - self.rest_twist[i, j])
            self.f_twist[i, j] += ti.Vector([f[0], f[1], f[2]])
            self.f_twist[i, j+1] += ti.Vector([f[4], f[5], f[6]])
            self.f_twist[i, j+2] += ti.Vector([f[8], f[9], f[10]])
            self.tau_twist[i, j] += f[3]
            self.tau_twist[i, j+1] += f[7]
    
    @ti.kernel
    def Compute_Gravity(self):
        for i, j in self.X:
            self.f_gravity[i, j] += self.m[i, j] * self.Gravity

    @ti.kernel
    def Explicit_X(self):
        for i, j in self.X:
            f = self.f_stretch[i, j] + self.f_bend[i, j] + self.f_twist[i, j] + self.f_gravity[i, j]
            if not self.Fixed[i, j]:
                self.V[i, j] += self.dt * (f / self.m[i, j])
                self.X[i, j] += self.dt * self.V[i, j]

    @ti.kernel
    def Explicit_Theta(self):
        for i, j in self.Theta:
            tau = self.tau_bend[i, j] + self.tau_twist[i, j]
            self.Omega[i, j] += self.dt * (tau / self.I[i, j])
            self.Theta[i, j] += self.dt * self.Omega[i, j]

    def Explicit_step(self, gravity=1):

        # update variables
        self.Update_tangent_related()
        self.Update_frame()
        self.Update_curvature()
        self.Update_twist()

        # calculate forces(generally)
        self.Clear()
        self.Compute_Stretch()
        self.Compute_DkappaDx()
        self.Compute_Bend_Gradient()
        self.Compute_DtwistDx()
        #self.Compute_Twist_Gradient()
        if gravity:
            self.Compute_Gravity()

        # explicit integration
        self.Explicit_X()
        self.Explicit_Theta()

    @ti.kernel
    def Interupt_Theta(self):
        for i in ti.static(range(self.num_rod)):
            self.Theta[i, self.num-2] += 10.0

    def write_to_file(self, outfile, frame):
        outfile.write("------frame {}-----\n".format(frame))
        outfile.write("position:\n{}\n".format(self.X))
        outfile.write("velocity:\n{}\n".format(self.V))
        outfile.write("omega:\n{}\n".format(self.Omega))
        outfile.write("theta:\n{}\n".format(self.Theta))
        outfile.write("ref_twist:\n{}\n".format(self.ref_twist))
        outfile.write("mass:\n{}\n".format(self.m))
        outfile.write("edge length:\n{}\n".format(self.edge_length))
        outfile.write("twist:\n{}\n".format(self.twist))
        outfile.write('streching force:\n{}\n'.format(self.f_stretch))
        outfile.write('bending force:\n{}\n'.format(self.f_bend))
        outfile.write('twisting force:\n{}\n'.format(self.f_twist))


        

    