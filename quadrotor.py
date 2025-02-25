import numpy as np
from torch import masked_scatter
from utils import *


class Quadrotor:
    def __init__(self):
        # Maximim thrust
        self.max_thrust =20

        # System state space
        self.pos = np.array([0,0,0])  #np.zeros((3,))
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  #Quaternion: qw, qx, qy, qz
        self.vel = np.array([0.0,0.0,0.0])
        self.a_rate = np.array([0.0,0.0,0.0])

        # Input constraints
        self.max_input_value =1.0
        self.min_input_value =0.0

        # Quadrotor intrinsic parameters
        self.J = np.array([0.03, 0.03, 0.06])  # Nm/s^2 = kg/m^2
        self.mass =1# 1kg
        self.mass_change =self.mass-1  # kg

        # Length of motor to CoG segment
        self.length = 0.47 / 2  # m

        # Positions of thrusters
        h = np.cos(np.pi / 4) * self.length
        self.x_f = np.array([h, -h, -h, h])
        self.y_f = np.array([-h, -h, h, h])

        # For z thrust torque calculation
        self.c = 0.013  # m   (z torque generated by each motor)
        self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

        # Gravity vector
        self.g = np.array([[0], [0], [9.81]])  # m s^-2

        # Actuation thrusts
        self.u = np.array([0.0, 0.0, 0.0, 0.0])  # N

        # Drag coefficients [kg / m]
        self.rotor_drag_xy = 0.3
        self.rotor_drag_z = 0.0  # No rotor drag in the z dimension
        self.rotor_drag = np.array(
            [self.rotor_drag_xy, self.rotor_drag_xy, self.rotor_drag_z]
        )[:, np.newaxis]
        self.aero_drag = 0.08

    def get_state(self):
        return [self.pos, self.quat, self.vel, self.a_rate]

    def get_control(self):
        return self.u

    def update(self, u, dt):
        """
        Euler dynamics integration

        :param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
        :param dt: time differential
        """

        # Clip inputs
        for i in range(u.shape[0]):
            u[i] = max(min(u[i], self.max_input_value), self.min_input_value)
        self.u = u * self.max_thrust

        # TODO: Generate disturbance forces / torques
        #f_d = np.random.rand(3, 1)
        f_d = np.zeros((3, 1))#np.ones((3,1))
        t_d = np.zeros((3, 1))

        x = self.get_state()

        # Euler integration
        self.pos = self.pos + dt*self.f_pos(x)
        quat = self.quat + dt*self.f_att(x)
        self.quat = unit_quat(quat)
        self.vel = self.vel + dt*self.f_vel(x, self.u, f_d)
        self.a_rate = self.a_rate + dt*self.f_rate(x, self.u, t_d)
    
    def f_pos(self, x):
        """
        Time-derivative of the position vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: position differential increment (vector): d[pos_x; pos_y]/dt
        """

        vel = x[2]
        return vel

    def f_att(self, x):
        """
        Time-derivative of the attitude in quaternion form
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
        """

        rate = x[3]
        angle_quaternion = x[1]

        return 1 / 2 * skew_symmetric(rate).dot(angle_quaternion)

    def f_vel(self, x, u, f_d):
        """
        Time-derivative of the velocity vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param f_d: disturbance force vector (3-dimensional)
        :return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
        """

        a_thrust = np.array([[0], [0], [np.sum(u)]]) / self.mass
        angle_quaternion = x[1]

        return np.squeeze(-self.g + v_dot_q(a_thrust + f_d / self.mass, angle_quaternion))

    def f_rate(self, x, u, t_d):
        """
        Time-derivative of the angular rate
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param t_d: disturbance torque (3D)
        :return: angular rate differential increment (scalar): dr/dt
        """

        rate = x[3]
        return np.array([
            ( u.dot(self.y_f)     + t_d[0] + (self.J[1] - self.J[2]) * rate[1] * rate[2])/self.J[0],
            (-u.dot(self.x_f)     + t_d[1] + (self.J[2] - self.J[0]) * rate[2] * rate[0])/self.J[1],
            ( u.dot(self.z_l_tau) + t_d[2] + (self.J[0] - self.J[1]) * rate[0] * rate[1])/self.J[2],
        ]).squeeze()
