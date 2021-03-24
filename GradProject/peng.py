from __future__ import print_function
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
    def __init__(self,n,times, mass=1,length=1, y0=None):

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
            # make new frame
            frame_i = frame.orientnew(f'A{i}', 'Axis', [q[i], frame.z])
            frame_i.set_ang_vel(frame, u[i] * frame.z)

            # make new point and have it rotate around previous point
            point_i = point.locatenew(f'P{i}', l[i] * frame_i.x)
            point_i.v2pt_theory(point, frame, frame_i)

            # make new particle at that point
            Pai = Particle(f'Pa{i}', point_i, m[i])
            bodies.append(Pai)

            # Set forces
            loads.append((point_i, m[i] * g * frame.x))
            kd_eqs.append(q[i].diff(t) - u[i])

            point = point_i

        # make equations
        method = KanesMethod(frame, q_ind=q, u_ind=u,                                   kd_eqs=kd_eqs)
        (fr, fr_star) = method.kanes_equations(bodies=bodies, loads=loads)
        kds = method.kindiffdict()

        # todo: make this dynamic
        if y0 is None:
            y0 =  [np.pi/4]*n + [0]*n  #inial position and velocities


        lengths = np.ones(n) /n # make all lengths sum to one
        print(f'y0 {y0}')
        print(f'length {lengths}')
        masses = [mass] * n
        print(f'masses {masses}')

        parameters = [g] + list(l) + list(m)
        par_val = [9.81] + list(lengths) + list(masses)
        print(f"par {parameters}")
        print(f'par_values {par_val}')


        dummy_symbols = [Dummy() for i in q + u]
        dummy_dict = dict(zip(q + u, dummy_symbols))


        # substitute unknown symbols for qdot terms
        mm_sym = method.mass_matrix_full.subs(kds).subs(dummy_dict)
        fo_sym = method.forcing_full.subs(kds).subs(dummy_dict)

        # create functions for numerical calculation
        mm_func = lambdify(dummy_symbols + parameters, mm_sym)
        fo_func = lambdify(dummy_symbols     + parameters, fo_sym)

        def f(y, t, args):
            vals = np.concatenate((y, args)) # combine variables
            dydt = np.array(np.linalg.solve(mm_func(*vals), fo_func(*vals))).T[0]  # solve mass matrix and force equations

            return dydt

        self.init_ode  = odeint(f, y0, times, args=(par_val,))
        temp = 0


def get_xy_coords(p):

    n = p.shape[1] // 2
    lengths = np.ones(n)/n
    zeros = np.zeros(p.shape[0])[:, None]
    x  = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])


    return np.cumsum(x, 1), np.cumsum(y, 1)

def GPE(y,n):
    mass =1
    points = y[:,1:]
    gpe=np.zeros((len(points),n))
    for i in range(n):
        gpe[:,i] = mass*9.8*(points[:,i]-(1/n*(i+1)))
    return gpe

def KE(ode_info,n):
    mass = 1
    ke_calc = np.square(ode_info[:,-n:])*mass
    return ke_calc

n=2
time = np.linspace(0,5,500)
app =n_pendulum(n,time,mass=1,length=1)
x_data, y_data = get_xy_coords(app.init_ode)
temp = GPE(y_data,n)
ke_temp =KE(app.init_ode,n)

kenergysum = np.sum(np.add(temp,ke_temp),axis=1)

# total_sum = np.sum(kenergysum)

# plt.plot(time,temp[:,0])
# plt.plot(time,temp[:,1])
# plt.plot(time,ke_temp[:,0])
# plt.plot(time,ke_temp[:,1])
#
# plt.show()

fig, ax = plt.subplots(2,2,  figsize=(8, 6))


ax[0,0].set_title(f'{n} pendulums ')
ax[0, 1].set_title('Potential Energy')
ax[1, 0].set_title('Kinetic Energy')
ax[1, 1].set_title('Total Energy')


ax[0,0].set_xlim(-1.2,1.2)
ax[0,0].set_ylim(-1.2,1.2)

pen_plot = ax[0,0].scatter(x_data[0], y_data[0], zorder = 5) #Pendilum plot


GPE_plot,  = ax[0, 1].plot([], [], lw=3)
KE_plot, = ax[1, 0].plot([], [], lw=3)

totalE_plot, = ax[1, 1].plot([], [], lw=3)
annotation=[]
for i in range(n):
    annotation.append(ax[0,0].annotate(f'  point {i+1}', xy=(x_data[0,i+1],y_data[0,i+1])))
arrow = []
for i in range(n):

    arrow.append(ax[0,0].annotate('', xy = (x_data[0,i], y_data[0,i]),
                        xytext = (x_data[0,i+1],y_data[0,i+1]),
                        arrowprops = {'arrowstyle': "-"}))

def animate(i) :
    pen_plot.set_offsets(np.c_[x_data[i,:], y_data[i,:]])
    tdata = time[:i]
    GPE_plot.set_data(tdata,temp[:i,:])
    KE_plot.set_data(tdata,ke_temp[:i,:])
    totalE_plot.set_data(tdata,kenergysum[:i])
    for j in range(1,n+1):
        annotation[j-1].set_position((x_data[i,j], y_data[i,j]))

    for j in range(n):
        start = (x_data[i,j], y_data[i,j])
        end   = (x_data[i,j+1], y_data[i,j+1])
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


ani = animation.FuncAnimation(fig, animate, frames=len(x_data),
                              interval = 10, blit = False)

plt.show()

