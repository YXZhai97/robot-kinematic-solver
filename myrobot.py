import numpy as np
import scipy.optimize



class MyRobot():

    def __init__(self, kinematic_chain, mass, joint_limits):
        self.tool = []
        self.world_frame = np.eye(4)
        self.home_position = np.array([20, 20, 20, 20, 20, 20])
        self.joints = self.home_position
        self.pastjoints = self.home_position
        self.joint_limits = joint_limits
        self.kinematic_chain = kinematic_chain
        self.mass = mass[0, :]
        self.com = mass[1:7, 0:3]
        self.dof = len(kinematic_chain)

    def __len__(self):
        return self.dof

    def transform(self, alpha, a, theta, d):

        sinalpha = np.sin(alpha)
        cosalpha = np.cos(alpha)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        t = np.array(
            [
                [costheta, -sintheta, 0, a],
                [sintheta * cosalpha, costheta * cosalpha, -sinalpha, -sinalpha * d],
                [sintheta * sinalpha, costheta * sinalpha, cosalpha, cosalpha * d],
                [0, 0, 0, 1],

            ]
        )

        return t

    def invtransform(self, alpha, a, theta, d):

        sinalpha = np.sin(alpha)
        cosalpha = np.cos(alpha)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        r = np.array(
            [
                [costheta, -sintheta, 0],
                [sintheta * cosalpha, costheta * cosalpha, -sinalpha],
                [sintheta * sinalpha, costheta * sinalpha, cosalpha],
            ]
        )

        t = np.array(
            [
                [a],
                [-sinalpha * d],
                [cosalpha * d],
            ]
        )

        invt = np.concatenate((r.T, -np.dot(r.T, t)), axis=1)
        invt = np.concatenate((invt, [[0, 0, 0, 1]]), axis=0)

        return invt

    def g_i(self, i):
        g_b = np.array([0, 0, -9.81, 0])
        t = np.eye(4)
        for j in range(i):
            alpha = self.kinematic_chain[i, 0]
            a = self.kinematic_chain[i, 1]
            theta = self.joints[i]
            d = self.kinematic_chain[i, 3]
            t = np.dot(self.invtransform(alpha, a, theta, d), t)
        g_i = np.dot(t, g_b)
        return g_i

    def get_torque(self, wrench):
        Moment = np.zeros((6, 3))
        Force = np.zeros((6, 3))
        force = np.array(wrench[:3])
        moment = np.array(wrench[-3:])
        m = self.mass
        com = self.com
        g_b = np.array([0, 0, -9.81, 0]);  # gravity in world coordinate
        M = np.zeros((7, 7, 3))
        F = np.zeros((7, 7, 3))
        M_g = np.zeros((7, 7, 3))
        for i in range(5, -1, -1):

            g_i = self.g_i(i)[:3]
            alpha = self.kinematic_chain[i, 0]
            a = self.kinematic_chain[i, 1]
            theta = self.joints[i]
            d = self.kinematic_chain[i, 3]
            transform = self.transform(alpha, a, theta, d)
            rotation = transform[:3, :3]
            if i == 5:
                F[i + 1, i + 1] = force
                M[i, i + 1] = moment
                F[i, i] = m[i] * g_i + F[i + 1, i + 1]
            else:
                F[i, i] = m[i] * g_i + np.dot(rotation, F[i + 1, i + 1]).T
                M[i, i + 1] = np.dot(rotation, M[i + 1, i + 1]) + np.cross(com[i], np.dot(rotation, F[i + 1, i + 1]))
            M_g[i, i] = m[i] * np.cross(self.com[i], g_i)
            M[i, i] = M_g[i, i] + M[i, i + 1]

            Moment[i] = M[i, i]
            Force[i] = F[i, i]

        return Moment, Force

    def matrix(self, q):

        t_world = np.eye(4)

        t = []
        for i in range(len(self)):
            alpha = self.kinematic_chain[i, 0]
            a = self.kinematic_chain[i, 1]
            theta = q[i]
            d = self.kinematic_chain[i, 3]
            t.append(self.transform(alpha, a, theta, d))

        t_tool = np.eye(4)

        return t

    def dq(self, q):
        # not finished yet
        return np.zeros(6)

    def fk(self, q):

        q = q + self.dq(q)
        self.pastjoints = self.joints
        self.joints = q
        q = np.deg2rad(q)
        t = np.eye(4)
        for i in range(len(self)):
            t = np.dot(t, self.matrix(q)[i])
        return t

    def ik(self, pose):
        x0 = self.home_position
        result = scipy.optimize.least_squares(
            fun=ik_cost_function, x0=x0, bounds=self.joint_limits, args=(pose, self)
        )  # type: scipy.optimize.OptimizeResult

        if result.success:
            actual_pose = self.fk(result.x)
            if np.allclose(actual_pose, pose, atol=1e-3):
                return result.x
        return None

    @classmethod
    def importrobot(cls, dhparameters, massparameters, jointlimits):
        """Construct Robot from parameters."""

        return cls(kinematic_chain=dhparameters, mass=massparameters, joint_limits=jointlimits)


def ik_cost_function(q, pose, robot):
    actual_pose = robot.fk(q)
    diff = np.abs(actual_pose - pose)
    # fun must return at most 1-d array_like.
    return diff.ravel()
