# from __future__ import print_function
from sympy import Dummy, lambdify
from scipy.integrate import odeint

import math
import numpy as np
import pylab as py

# import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import pyplot as plt

from sympy.physics.mechanics import *
import sympy as sy


class n_pendulum:
    def __init__(self, n, times, mass=1, y0=None, position= None):

        self.n = n

        self.lengths = np.ones(n) / n  # make all lengths sum to one
        self.masses = [mass] * n

        # dynamic vars
        q = dynamicsymbols(f"q:{n}")  # coordinates
        u = dynamicsymbols(f"u:{n}")  # speeds

        m = sy.symbols(f"m:{n}")  # Mass
        l = sy.symbols(f"l:{n}")  # Length
        g, t = sy.symbols("g, t")  # Gravity, time

        # Init model
        frame = ReferenceFrame("A")  # referance frame
        point = Point("P")  # pivot point
        point.set_vel(frame, 0)  # set velocity of pivot point

        bodies = []
        loads = []
        kd_eqs = []

        for i in range(n):
            # make new frame
            frame_i = frame.orientnew(f"A{i}", "Axis", [q[i], frame.z])
            frame_i.set_ang_vel(frame, u[i] * frame.z)

            # make new point and have it rotate around previous point
            point_i = point.locatenew(f"P{i}", l[i] * frame_i.x)
            point_i.v2pt_theory(point, frame, frame_i)

            # make new particle at that point
            Pai = Particle(f"Pa{i}", point_i, m[i])
            bodies.append(Pai)

            # Set forces
            loads.append((point_i, m[i] * g * frame.x))
            kd_eqs.append(q[i].diff(t) - u[i])

            point = point_i

        # make equations
        method = KanesMethod(frame, q_ind=q, u_ind=u, kd_eqs=kd_eqs)

        (fr, fr_star) = method.kanes_equations(bodies=bodies, loads=loads)
        print("\n\n\nRHS")
        for t in method.rhs("GE"):
            print(t)
        print()
        kds = method.kindiffdict()
        for t in kds:
            print(t)
        print("\n\n")
        print(fr)
        print(fr_star)

        if y0 is None:
            if position is None:
                y0 = [4 * np.pi / 8] * n + [0] * n  # inial position and velocities
            else:
                y0 = [position] *n + [0]*n
        parameters = [g] + list(l) + list(m)
        par_val = [1] + list(self.lengths) + list(self.masses)

        print(f"y0 {y0}")
        print(f"length {self.lengths}")
        print(f"masses {self.masses}")
        print(f"par {parameters}")
        print(f"par_values {par_val}")

        dummy_symbols = [Dummy() for i in q + u]
        dummy_dict = dict(zip(q + u, dummy_symbols))

        print("dummy dict", dummy_dict)
        # substitute unknown symbols for qdot terms
        mm_sym = method.mass_matrix_full.subs(kds).subs(dummy_dict)
        fo_sym = method.forcing_full.subs(kds).subs(dummy_dict)
        print(f"mass matrix {mm_sym}")
        print(f"force vectors {fo_sym}")

        # create functions for numerical calculation
        mm_func = lambdify(
            dummy_symbols + parameters, mm_sym
        )  #  function taking in symbols and peramitors
        fo_func = lambdify(
            dummy_symbols + parameters, fo_sym
        )  # force function taking in symbols and peramitors

        def f(y, t, args):
            vals = np.concatenate((y, args))  # combine variables
            dydt = np.array(np.linalg.solve(mm_func(*vals), fo_func(*vals))).T[
                0
            ]  # solve mass matrix and force equations

            return dydt

        ##double check rubrick###
        self.init_ode = odeint(f, y0, times, args=(par_val,))

    def get_xy_coords(
        self,
    ):
        p = self.init_ode
        zeros = np.zeros(self.init_ode.shape[0])[:, None]
        x = np.hstack([zeros, self.lengths * np.sin(p[:, : self.n])])
        y = np.hstack([zeros, -self.lengths * np.cos(p[:, : self.n])])
        return np.cumsum(x, 1), np.cumsum(y, 1)

    def GPE(self):

        gpe = np.zeros((len(self.init_ode), self.n))
        thetas = self.init_ode[:, : self.n]

        for k in range(self.n):
            cos_values = 0
            for i in range(k + 1):
                cos_values += np.cos(thetas[:, i])
            cos_values *= self.masses[k] * self.lengths[k]
            gpe[:, k] += cos_values

        return gpe

    def KE(self):
        theta_dot = self.init_ode[:, -self.n :]
        theta = self.init_ode[:, : self.n]
        ke_calc = np.zeros((len(self.init_ode), n))

        print("KE Formula")
        for k in range(n):
            k_values = 0
            print(f"k{k} =  1/2 * mass[{k}] * (", end="")
            for i in range(k + 1):
                for j in range(i, k + 1):
                    if i == j:
                        print(f"lengths[{i}]^2 * theta_dot[:, {i}]^2", end=" + ")
                        k_values += np.power(self.lengths[i], 2) * np.power(
                            theta_dot[:, i], 2
                        )
                    else:
                        print(
                            f" 2*lengths[{i}]*lengths[{j}]*theta_dot[:,{i}]*theta_dot[:,{j}]*(np.cos(theta[:,{i}] - theta[:,{j}])"
                        )
                        k_values += (
                            2
                            * self.lengths[i]
                            * self.lengths[j]
                            * (np.cos(theta[:, i] - theta[:, j]))
                            * theta_dot[:, i]
                            * theta_dot[:, j]
                        )
            k_values *= 0.5 * self.masses[k]
            ke_calc[:, k] += k_values
            print(")")

        return ke_calc


n = 3


time = np.linspace(0, 20, 1000)
app = n_pendulum(n, time, mass=1)
x_data, y_data = app.get_xy_coords()
gpe_values = app.GPE()
ke_values = app.KE()

total_energy = np.sum(np.subtract(ke_values, gpe_values), axis=1)


fig, ax = plt.subplots(2, 2, figsize=(8, 8))


ax[0, 0].set_title(f"{n} pendulums ")
ax[0, 1].set_title("Potential Energy")
ax[1, 0].set_title("Kinetic Energy")
ax[1, 1].set_title("Total Energy")


ax[0, 0].set_xlim(-1.2, 1.2)
ax[0, 0].set_ylim(-1.2, 1.2)

pen_plot = ax[0, 0].scatter(x_data[0, 0], y_data[0, 0], zorder=5)  # Pendilum plot


gpe_list = []
ke_list = []
for i in range(n):
    (GPE_plot,) = ax[0, 1].plot([], [], lw=3)
    (KE_plot,) = ax[1, 0].plot([], [], lw=3)
    gpe_list.append(GPE_plot)
    ke_list.append(KE_plot)

(totalE_plot,) = ax[1, 1].plot([], [], lw=3)
annotation = []
for i in range(n):
    annotation.append(
        ax[0, 0].annotate(f"  point {i+1}", xy=(x_data[0, i + 1], y_data[0, i + 1]))
    )
arrow = []
for i in range(n):
    arrow.append(
        ax[0, 0].annotate(
            "",
            xy=(x_data[0, i], y_data[0, i]),
            xytext=(x_data[0, i + 1], y_data[0, i + 1]),
            arrowprops={"arrowstyle": "-"},
        )
    )


def animate(i):
    pen_plot.set_offsets(np.c_[x_data[i, :], y_data[i, :]])
    tdata = time[:i]
    for j in range(n):
        gpe_list[j].set_data(tdata, gpe_values[:i, j])
        ke_list[j].set_data(tdata, ke_values[:i, j])
    totalE_plot.set_data(tdata, total_energy[:i])
    for j in range(1, n + 1):
        annotation[j - 1].set_position((x_data[i, j], y_data[i, j]))

    for j in range(n):
        start = (x_data[i, j], y_data[i, j])
        end = (x_data[i, j + 1], y_data[i, j + 1])
        arrow[j].set_position(start)
        arrow[j].xy = end

    ax[0, 0].relim()
    ax[0, 0].autoscale_view(True, True, True)
    ax[0, 1].relim()
    ax[0, 1].autoscale_view(True, True, True)

    ax[1, 0].relim()
    ax[1, 0].autoscale_view(True, True, True)
    ax[1, 1].relim()
    ax[1, 1].autoscale_view(True, True, True)


ani = animation.FuncAnimation(fig, animate, frames=len(x_data), interval=1, blit=False)

plt.show()
from IPython.display import HTML
# HTML(ani.to_html5_video())
from matplotlib import collections

def animate_pendulum_multiple(n, number_of_pendulums=20, perturbation=1E-3, track_length=15):
    oversample = 3
    track_length *= oversample

    t = np.linspace(0, 20, oversample * 200)
    p = [n_pendulum(n, t, position=135 + i * perturbation / number_of_pendulums)
         for i in range(number_of_pendulums)]
    positions = np.array([pi.get_xy_coords() for pi in p])
    positions = positions.transpose(0, 2, 3, 1)
    # positions is a 4D array: (npendulums, len(t), n+1, xy)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))

    track_segments = np.zeros((number_of_pendulums, 0, 2))
    tracks = collections.LineCollection(track_segments, cmap='rainbow')
    tracks.set_array(np.linspace(0, 1, number_of_pendulums))
    ax.add_collection(tracks)

    points, = plt.plot([], [], 'ok')

    pendulum_segments = np.zeros((number_of_pendulums, 0, 2))
    pendulums = collections.LineCollection(pendulum_segments, colors='black')
    ax.add_collection(pendulums)

    def init():
        pendulums.set_segments(np.zeros((number_of_pendulums, 0, 2)))
        tracks.set_segments(np.zeros((number_of_pendulums, 0, 2)))
        points.set_data([], [])
        return pendulums, tracks, points

    def animate(i):
        i = i * oversample
        pendulums.set_segments(positions[:, i])
        sl = slice(max(0, i - track_length), i)
        tracks.set_segments(positions[:, sl, -1])
        x, y = positions[:, i].reshape(-1, 2).T
        points.set_data(x, y)
        return pendulums, tracks, points

    interval = 1000 * oversample * t.max() / len(t)
    anim = animation.FuncAnimation(fig, animate, frames=len(t) // oversample,
                                   interval=interval,
                                   blit=True, init_func=init)
    plt.show()
    plt.close(fig)
    return anim


# anim = animate_pendulum_multiple(2)
# HTML(anim.to_html5_video())