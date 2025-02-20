import os
import casadi as ca
import numpy as np
from quadrotor import Quadrotor
from utils import skew_symmetric, v_dot_q, quaternion_inverse
from nn_casadi_change_value_function import pytorch_to_casadi
from nn_casadi_change_sensitivity import pytorch_to_casadi_sensitivity
import pickle


class Controller_nn:
    def __init__(self, quad: Quadrotor, obstacles=None, n_nodes=30, dt=0.1, use_rbf=False, gamma=0.9, use_nn=False, model_path='value_function_model.pth',use_sensitivity=False, model_path_sensitivity='neural_sensitivity_model.pth'):
        """
        :param quad: quadrotor object
        :param obstacles: List of obstacles, each defined by (position, radius)
        :param n_nodes: number of optimization nodes until time horizon
        :param dt: time step
        :param use_rbf: whether to use RBF network for prediction
        :param gamma: CBF parameter
        :param use_nn: whether to integrate neural network for cost prediction
        :param model_path: path to the PyTorch model for cost prediction
        gamma=0.1
        """
        self.N = n_nodes    # number of control nodes within horizon
        self.dt = dt        # time step
        self.x_dim = 13
        self.u_dim = 4
        self.use_rbf = use_rbf
        self.gamma = gamma  # CBF parameter
        self.use_nn = use_nn  # Whether to use neural network for cost prediction
        self.use_sensitivity = use_sensitivity


        self.quad_radius = 0.5  
        if obstacles is None:
            obstacles = []  
        self.obstacles = obstacles


        self.opti = ca.Opti()
        self.opt_states = self.opti.variable(self.x_dim, self.N + 1)
        self.opt_controls = self.opti.variable(self.u_dim, self.N)

        self.quad = quad
        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        # Declare model variables
        self.p = self.opt_states[:3, :]      # position
        self.q = self.opt_states[3:7, :]     # angle quaternion (wxyz)
        self.v = self.opt_states[7:10, :]    # velocity
        self.r = self.opt_states[10:13, :]   # angular rate

        f = lambda x_, u_, f_d_: ca.vertcat(*[
            self.p_dynamics(x_),
            self.q_dynamics(x_),
            self.v_dynamics(x_, u_, f_d_),
            self.w_dynamics(x_, u_)
        ])

        # Noise variables
        self.f_d = self.opti.parameter(3, 1)
        self.f_t = self.opti.parameter(3, 1)

        # Initial condition
        self.opt_x_ref = self.opti.parameter(self.x_dim, self.N + 1)
        self.opti.subject_to(self.opt_states[:, 0] == self.opt_x_ref[:, 0])
        for i in range(self.N):
            x_next = self.opt_states[:, i] + f(self.opt_states[:, i], self.opt_controls[:, i], self.f_d) * self.dt
            self.opti.subject_to(self.opt_states[:, i + 1] == x_next)

        # Weighted squared error loss function
        q_cost = np.diag([6, 5, 5, 0.1, 0.1, 0.1, 0.1, 5, 5, 5, 5, 5, 5])
        r_cost = np.diag([0.1, 0.1, 0.1, 0.1])

        # Cost function
        self.obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[:, i] - self.opt_x_ref[:, i + 1]
            self.obj += ca.mtimes([state_error_.T, q_cost, state_error_]) \
                       + ca.mtimes([self.opt_controls[:, i].T, r_cost, self.opt_controls[:, i]])

        # Integrate neural network if enabled
        if self.use_nn:
            # Load neural network from PyTorch model and convert to CasADi function
            nn_function = pytorch_to_casadi(model_path)  # 加载神经网络
            # Use the state at time step 5 to predict the cost
            x_ref_5 = self.opt_states[:, 5]  # Get the 5th step state
            cost_nn = nn_function(x_ref_5)  # Neural network cost for the next 25 steps
            self.obj += cost_nn  # Add the neural network's cost to the objective


        if self.use_sensitivity:
            # Load neural network from PyTorch model and convert to CasADi function
            nn_sensitivity = pytorch_to_casadi_sensitivity(model_path_sensitivity)  # 加载神经网络
            # Use the state at time step 5 to predict the cost
            x_ref_5 = self.opt_states[:, 5]  # Get the 5th step state
            cost_nn_sensitivity = nn_sensitivity(x_ref_5)  # Neural network cost for the next 25 steps
            self.obj = self.obj + cost_nn_sensitivity*self.quad.mass_change  # Add the neural network's cost to the objective           

        # Add slack variable penalty if using soft constraints
        if self.use_rbf:
            # Ensure slack variables are defined
            self.slack = self.opti.variable(len(self.obstacles), self.N + 1)
            self.opti.subject_to(self.slack >= 0)
            # Example penalty, adjust as needed
            self.obj += 1000 * ca.sumsqr(self.slack)

        self.opti.minimize(self.obj)

        # Control input constraints
        self.add_input_constraints()

        # Add obstacle constraints over the entire prediction horizon
        delta = 0.5  # Adjust safety threshold as needed
        #self.add_obstacle_constraints(delta)

        # Add CBF constraints
        self.add_cbf_constraints()

        # Solver options
        opts_setting = {
            'ipopt.max_iter': 500000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-7,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        self.opti.solver('ipopt', opts_setting)

    def add_input_constraints(self):

        for j in range(self.u_dim):
            self.opti.subject_to(self.opti.bounded(self.min_u, self.opt_controls[j, :], self.max_u))

    def add_obstacle_constraints(self, delta):

        for idx, (p, r_obstacle) in enumerate(self.obstacles):

            h = ca.sum1((self.p - p[:, None]) ** 2) - (r_obstacle + self.quad_radius) ** 2 - delta
            if self.use_rbf:

                self.opti.subject_to(h >= -self.slack[idx, :])
            else:

                self.opti.subject_to(h >= 0)

    def add_cbf_constraints(self):

        if not self.obstacles:
            return  

        for idx, (p, r_obstacle) in enumerate(self.obstacles):
            for k in range(self.N):

                H_k = ca.sum1((self.p[:, k] - p) ** 2) - (r_obstacle + self.quad_radius) ** 2
                H_k1 = ca.sum1((self.p[:, k + 1] - p) ** 2) - (r_obstacle + self.quad_radius) ** 2


                self.opti.subject_to(H_k1 >= (1 - self.gamma) * H_k)

    def p_dynamics(self, x):
        """
        Time-derivative of the position vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: position differential increment (vector): d[pos_x; pos_y]/dt
        """
        vel = x[7:10]
        return vel

    def q_dynamics(self, x):
        """
        Time-derivative of the attitude in quaternion form
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
        """
        q = x[3:7]
        r = x[10:13]
        return 0.5 * ca.mtimes(skew_symmetric(r), q)

    def v_dynamics(self, x, u, f_d):
        """
        Time-derivative of the velocity vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param f_d: disturbance force vector (3-dimensional)
        :return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
        """
        q = x[3:7]
        f_thrust = u * self.quad.max_thrust
        g = ca.vertcat(0.0, 0.0, 9.81)
        a_thrust = (ca.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / self.quad.mass) + \
                   (f_d / self.quad.mass)

        v_dynamics = v_dot_q(a_thrust, q) - g
        return v_dynamics

    def w_dynamics(self, x, u):
        """
        Time-derivative of the angular rate
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :return: angular rate differential increment (scalar): dr/dt
        """
        r = x[10:13]
        f_thrust = u * self.quad.max_thrust

        y_f = ca.MX(self.quad.y_f)
        x_f = ca.MX(self.quad.x_f)
        c_f = ca.MX(self.quad.z_l_tau)
        return ca.vertcat(
            (ca.mtimes(f_thrust.T, y_f) + (self.quad.J[1] - self.quad.J[2]) * r[1] * r[2]) / self.quad.J[0],
            (-ca.mtimes(f_thrust.T, x_f) + (self.quad.J[2] - self.quad.J[0]) * r[2] * r[0]) / self.quad.J[1],
            (ca.mtimes(f_thrust.T, c_f) + (self.quad.J[0] - self.quad.J[1]) * r[0] * r[1]) / self.quad.J[2]
        )

    def compute_control_signal(self, x_ref):

        # RBF net prediction
        if self.use_rbf:
            e_pos = (x_ref[:3, 1] - self.quad.pos)
            e_vel = (x_ref[7:10, 1] - self.quad.vel)
            error = 3 * e_pos + e_vel
            f_d = self.rbfnet.predict(self.quad.pos, self.quad.vel, error=error)
            print("Error: {}, predict: {}".format(error, f_d))
        else:
            f_d = np.zeros((3, 1))

        # Set parameters
        self.opti.set_value(self.opt_x_ref, x_ref)
        self.opti.set_value(self.f_d, f_d)

        try:
            sol = self.opti.solve()
        except RuntimeError:
            print("Optimization failed, returning zero control input.")
            return np.zeros(self.u_dim), None, None, None 

        u = sol.value(self.opt_controls)
        cost_over_30 = sol.value(self.obj) 

   
        q_cost = np.diag([5, 5, 5, 0.1, 0.1, 0.1, 0.1, 5, 5, 5, 5, 5, 5])
        r_cost = np.diag([0.1, 0.1, 0.1, 0.1])
        cost_over_last_25 = 0
        for i in range(5, self.N):
            state_error_ = self.opt_states[:, i] - self.opt_x_ref[:, i + 1]
            cost_over_last_25 += ca.mtimes([state_error_.T, q_cost, state_error_]) \
                                 + ca.mtimes([self.opt_controls[:, i].T, r_cost, self.opt_controls[:, i]])

        cost_over_last_25 = sol.value(cost_over_last_25)

        state_step_5 = sol.value(self.opt_states[:, 5])

        
        return u[:, 0], cost_over_30, cost_over_last_25, state_step_5
