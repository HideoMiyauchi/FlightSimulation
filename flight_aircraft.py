# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

class FlightAircraft:

    # event handler for 2-windows coordination
    def on_move(self, event):
        if event.inaxes == self.axes1:
            self.axes2.view_init(elev=self.axes1.elev, azim=self.axes1.azim)
        elif event.inaxes == self.axes2:
            self.axes1.view_init(elev=self.axes2.elev, azim=self.axes2.azim)
        else:
            return
        self.fig.canvas.draw_idle()

    def __init__(self, width=4, height=4, scale=1, persist=False):
        pyplot.rcParams["font.size"] = 6

        # aircraft's vertex
        self.aircraft_vertex = np.array([
            [30, 1, 0], [-30, 4, 0], [-30, -4, 0], [30, -1, 0],
            [-15, 2, -8], [-15, -2, -8], [5, 4, -4], [-10, 27, -4],
            [-15, 20, -4], [-5, 4, -4], [5, -4, -4], [-10, -27, -4],
            [-15, -20, -4], [-5, -4, -4]
        ])

        # create big window
        self.fig = pyplot.figure(figsize=(width, height))
        self.axes1 = pyplot.axes(projection="3d")
        self.axes1.invert_zaxis() # upside down
        self.scale = scale
        self.persist = persist # flag to erase what was previously drawn
        self.axes1_maxx = -9999
        self.axes1_maxy = -9999
        self.axes1_maxz = -9999
        self.axes1_minx = 9999
        self.axes1_miny = 9999
        self.axes1_minz = 9999

        # create small window
        self.axes2 = pyplot.axes([0, 0, 0.3, 0.3], projection="3d")
        self.axes2.tick_params(labelbottom=False, labelleft=False,
            labelright=False, labeltop=False,
            bottom=False, left=False, right=False, top=False)
        self.axes2.invert_zaxis() # upside down
        self.axes2.grid(True)
        self.axes2.set_xlim(-30, 30)
        self.axes2.set_ylim(-30, 30)
        self.axes2.set_zlim(-34, 26)

        # entry event handler
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    # 3-axis rotate
    def get_rotate_matrix(self, phi, theta, psi):
        rx = np.array([
            [np.cos(psi), np.sin(psi), 0],
            [-np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        ry = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])
        rz = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        return rx.dot(ry.dot(rz))

    # draw big window
    def draw_axes1(self, x, y, z, phi, theta, psi):

        # expand scale objects
        vpos = np.dot(self.aircraft_vertex, self.scale)

        # rotate objects
        rotate_matrix = self.get_rotate_matrix(phi, theta, psi)
        vpos = np.dot(rotate_matrix, vpos.T).T

        # translate objects
        shift = np.tile(np.array([x,-y,z]), (vpos.shape[0],1))
        vpos = vpos + shift

        # clear window
        if self.persist == False:
            for artist in self.axes1.lines + self.axes1.collections:
                artist.remove()

        # draw
        p = np.empty((0,3), int)
        for i in [1,2]:
            p = np.append(p, [vpos[i,:]], axis=0)
        self.axes1.plot(p[:,0], p[:,1], p[:,2], linestyle="solid",
            linewidth=4.0, color="red")

        p = np.empty((0,3), int)
        for i in [4,5]:
            p = np.append(p, [vpos[i,:]], axis=0)
        self.axes1.plot(p[:,0], p[:,1], p[:,2], linestyle="solid",
            linewidth=4.0, color="blue")

        p = np.empty((0,3), int)
        for i in [9,8,7,6,0,1,4,0,3,2,5,3,10,11,12,13]:
            p = np.append(p, [vpos[i,:]], axis=0)
        self.axes1.plot(p[:,0], p[:,1], p[:,2], linestyle="solid",
            linewidth=0.4, color="black")

        # adjustment three axis
        self.axes1_maxx = max(self.axes1_maxx, max(p[:,0]))
        self.axes1_minx = min(self.axes1_minx, min(p[:,0]))
        self.axes1_maxy = max(self.axes1_maxy, max(p[:,1]))
        self.axes1_miny = min(self.axes1_miny, min(p[:,1]))
        self.axes1_maxz = max(self.axes1_maxz, max(p[:,2]))
        self.axes1_minz = min(self.axes1_minz, min(p[:,2]))
        midx = 0.5 * (self.axes1_maxx + self.axes1_minx)
        midy = 0.5 * (self.axes1_maxy + self.axes1_miny)
        midz = 0.5 * (self.axes1_maxz + self.axes1_minz)
        max_range = 0.5 * max(self.axes1_maxx - self.axes1_minx,
            self.axes1_maxy - self.axes1_miny,
            self.axes1_maxz - self.axes1_minz
        )
        self.axes1.set_xlim(midx - max_range, midx + max_range)
        self.axes1.set_ylim(midy - max_range, midy + max_range)
        self.axes1.set_zlim(midz - max_range, midz + max_range)

    # draw small window
    def draw_axes2(self, phi, theta, psi):

        # rotate objects
        rotate_matrix = self.get_rotate_matrix(phi, theta, psi)
        vpos = np.dot(rotate_matrix, self.aircraft_vertex.T).T

        # clear window
        for artist in self.axes2.lines + self.axes2.collections:
            artist.remove()

        # draw
        p = np.empty((0,3), int)
        for i in [1,2]:
            p = np.append(p, [vpos[i,:]], axis=0)
        self.axes2.plot(p[:,0], p[:,1], p[:,2], linestyle="solid",
            linewidth=4.0, color="red")

        p = np.empty((0,3), int)
        for i in [4,5]:
            p = np.append(p, [vpos[i,:]], axis=0)
        self.axes2.plot(p[:,0], p[:,1], p[:,2], linestyle="solid",
            linewidth=4.0, color="blue")

        p = np.empty((0,3), int)
        for i in [9,8,7,6,0,1,4,0,3,2,5,3,10,11,12,13]:
            p = np.append(p, [vpos[i,:]], axis=0)
        self.axes2.plot(p[:,0], p[:,1], p[:,2], linestyle="solid",
            linewidth=0.4, color="black")

    # entry
    def draw(self, x, y, z, phi, theta, psi):
        self.draw_axes1(x,y,z,phi,theta,psi)
        self.draw_axes2(phi,theta,psi)
