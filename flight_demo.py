# -*- coding: utf-8 -*-

import flight_aircraft
import numpy as np
import math
import signal
from matplotlib import pyplot

class FlightDynamics:

    def __init__(self):
        self.tick = 0.1 # 0.1(sec)
        self.h0 = 1500
        self.v0 = 100
        self.theta0 = 7.5
        self.kn = np.zeros(12)

    def diff_equation(self, elevator, aileron, throttle):
        (v, alpha, beta, p, q, r, phi, theta, psi, dmy, dmy, dmy) = self.kn

        # vertical equation
        self.v_equation = np.array([
            [-0.0293, -0.1059, 0.0, -0.1696, 0.0, 0.0000720],
            [-0.0961, -0.7158, 1.0, -0.0121, -0.1225, -0.0],
            [-0.0023, -0.99, -0.6939, -0.000289, -0.8066, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ])
        v_result = self.v_equation.dot(
            np.array([v, alpha, q, theta, elevator, throttle])
        )

        # horizontal equation
        self.h_equation = np.array([
            [-0.0487, 0.1309, -1.0, 0.0919, 0.0],
            [-4.8504, -1.6374, -0.2038, 0.0, -2.6708],
            [0.8636, -0.1498, -0.1919, 0.0, 0.1205],
            [0.0, 1.0, 0.1317, 0.0, 0.0]
        ])
        h_result = self.h_equation.dot(np.array([beta, p, r, phi, aileron]))

        # rotate equation
        r_equation = np.array([
            [ dcos(theta) * dcos(psi),
              dsin(phi) * dsin(theta) * dcos(psi) - dcos(phi) * dsin(psi),
              dcos(phi) * dsin(theta) * dcos(psi) + dsin(phi) * dsin(psi)
            ],
            [ dcos(theta) * dsin(psi),
              dsin(phi) * dsin(theta) * dsin(psi) + dcos(phi) * dcos(psi),
              dcos(phi) * dsin(theta) * dsin(psi) - dsin(phi) * dcos(psi)
            ],
            [ dsin(theta),
              -dsin(phi) * dcos(theta),
              -dcos(phi) * dcos(theta)
            ]
        ])
        r_param = np.array([
            (v + self.v0) * dcos(beta) * dcos(alpha),
            (v + self.v0) * dsin(beta),
            (v + self.v0) * dcos(beta) * dsin(alpha)
        ])
        r_result = r_equation.dot(r_param)

        return np.array([
            v_result[0], # v
            v_result[1], h_result[0], # alpha, beta
            h_result[1], v_result[2], h_result[2], # p, q, r
            h_result[3], v_result[3], r / dcos(self.theta0), # phi, theta, psi
            r_result[0], r_result[1], r_result[2] # x, y, z
        ])

    def solve(self, elevator, aileron, throttle):
        self.kn += self.tick * self.diff_equation(elevator, aileron, throttle)
        return (
            self.kn[9], self.kn[10], self.kn[11] + self.h0, # x, y, z
            self.kn[6], self.kn[7] + self.theta0, self.kn[8] # phi, theta, psi
        )

    def stop(self):
        pyplot.show()

    def keep_roll_angle(self, phi):
        kphi = 1.0
        kp = 3.0
        j1 = kphi * (phi - self.kn[6])
        j2 = kp * self.kn[3]
        return -j1 + j2

    def keep_pitch_angle(self, theta):
        ktheta = 1
        kq = 2
        j1 = ktheta * (theta - self.kn[7])
        j2 = kq * (-j1 + self.kn[4])
        return j2

    def keep_velocity(self):
        kthrottle = -10000
        j1 = kthrottle * self.kn[0]
        return j1

def dcos(v):
    return np.cos(np.radians(v))

def dsin(v):
    return np.sin(np.radians(v))

if __name__=="__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # create instance
    fd = FlightDynamics()
    fa = flight_aircraft.FlightAircraft(8, 8, scale=5, persist=True)

    for t in range(1300): # increments of seconds until 130(sec)

        # flight scenario
        if (t < 300):
            pitch = 30
            roll = 20
        else:
            pitch = -10
            roll = 50

        # auto pilot control
        elevator = fd.keep_pitch_angle(pitch)
        aileron = fd.keep_roll_angle(roll)
        throttle = fd.keep_velocity()

        (x, y, z, phi, theta, psi) = fd.solve(elevator, aileron, throttle)

        # draw aircraft
        if ((t % 50) == 0):
            fa.draw(x, y, z,
                np.radians(phi), np.radians(theta), np.radians(psi))
            pyplot.pause(0.1)

    fd.stop()


