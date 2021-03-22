from __future__ import print_function
from sympy import Dummy, lambdify
from scipy.integrate import odeint

import time
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
    def __init__(self,n,times, mass=1,length=1):

        # dynamic vars
        q = dynamicsymbols(f'q:{n}')  # coordinates
        u = dynamicsymbols(f'u:{n}')  # speeds

        m = sy.symbols(f'm:{n}')  # Mass
        l = sy.symbols(f'l:{n}')  # Length
        g, t = sy.symbols('g, t')  # Gravity, time

        # Init model
        frame = ReferenceFrame('A')  # referance frame
        point = Point('P')  # pivot point
        point.set_vel(frame, 0)  # set velocity of pivot point

        bodies = []
        loads = []
        kd_eqs = []

        for i in range(n):
            # Create a reference frame following the i^th mass
            frame_i = frame.orientnew('A' + str(i), 'Axis', [q[i], frame.z])
            frame_i.set_ang_vel(frame, u[i] * frame.z)

            # Create a point in this reference frame
            point_i = point.locatenew('P' + str(i), l[i] * frame_i.x)
            point_i.v2pt_theory(point, frame, frame_i)

            # Create a new particle of mass m[i] at this point
            Pai = Particle('Pa' + str(i), point_i, m[i])
            bodies.append(Pai)

            # Set forces & compute kinematic ODE
            loads.append((point_i, m[i] * g * frame.x))
            kd_eqs.append(q[i].diff(t) - u[i])

            point = point_i

        # Generate equations of motion
        method = KanesMethod(frame, q_ind=q, u_ind=u,                                   kd_eqs=kd_eqs)
        (fr, fr_star) = method.kanes_equations(bodies=bodies, loads=loads)

        y0 =  [2*np.pi/4,np.pi/4,np.pi/4,20,-10,20]  #inial position and velocities
        lengths = np.ones(n) /n

        lengths = np.broadcast_to(lengths, n)
        masses = np.broadcast_to(mass, n)

        # Fixed parameters: gravitational constant, lengths, and masses
        parameters = [g] + list(l) + list(m)
        par_val = [9.81] + list(lengths) + list(masses)
        print(f"par {parameters}")
        print(f'par_values {par_val}')


        dummy_symbols = [Dummy() for i in q + u]  # Create a dummy symbol for each variable
        dummy_dict = dict(zip(q + u, dummy_symbols))
        kds = method.kindiffdict()

        # substitute unknown symbols for qdot terms
        mm_sym = method.mass_matrix_full.subs(kds).subs(dummy_dict)
        fo_sym = method.forcing_full.subs(kds).subs(dummy_dict)

        # create functions for numerical calculation
        mm_func = lambdify(dummy_symbols + parameters, mm_sym)
        fo_func = lambdify(dummy_symbols     + parameters, fo_sym)

        def f(y, t, args):
            vals = np.concatenate((y, args))
            sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
            return np.array(sol).T[0]
        self.init_ode  = odeint(f, y0, times, args=(par_val,))

def get_xy_coords(p, lengths):
    """Get (x, y) coordinates from generalized coordinates p"""
    p = np.atleast_2d(p)
    n = p.shape[1] // 2
    lengths = np.ones(n)/n
    zeros = np.zeros(p.shape[0])[:, None]
    x  = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])


    return np.cumsum(x, 1), np.cumsum(y, 1)

time = np.linspace(0,5,500)

app =n_pendulum(3,time,mass=1,length=1)

x_data, y_data = get_xy_coords(app.init_ode,1)

fig, ax = plt.subplots(figsize = (8,6))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

scatter = ax.scatter(x_data[0], y_data[0], zorder = 5) #Scatter plot

annotation = ax.annotate('  point 1', xy=(x_data[0,1],y_data[0,1]))

arrow = ax.annotate('', xy = (x_data[0,0], y_data[0,0]),
                        xytext = (x_data[0,1],y_data[0,1]),
                        arrowprops = {'arrowstyle': "-"})

def animate(i) :
    scatter.set_offsets(np.c_[x_data[i,:], y_data[i,:]])

    annotation.set_position((x_data[i,1], y_data[i,1]))

    start = (x_data[i,0], y_data[i,0])
    end   = (x_data[i,1], y_data[i,1])
    arrow.set_position(start)
    arrow.xy = end

ani = animation.FuncAnimation(fig, animate, frames=len(x_data),
                              interval = 10, blit = False)

plt.show()

