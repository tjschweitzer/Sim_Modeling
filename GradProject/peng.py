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





# Genralized Variables
n = 1
q = dynamicsymbols(f'q:{n}')  # coordinates
u = dynamicsymbols(f'u:{n}')  # speeds

m = sy.symbols(f'm:{n}')  # Mass
l = sy.symbols(f'l:{n}')  # Length

g, t = sy.symbols('g, t')  # Gravity, time

time =np.linspace(0, 10, 1000)


# Init model
frame = ReferenceFrame('A')  # referance frame
point = Point('P')  # pivot point
point.set_vel(frame, 0)  # set velocity to pivot point to 0


# Create a reference frame following the i^th mass
frame_0 = frame.orientnew('A0', 'Axis', [q[0], frame.z])
frame_0.set_ang_vel(frame, u[0] * frame_0.z)

# Create a point in this reference frame
point0 = point.locatenew('P0', l[0] * frame_0.x)
point0.v2pt_theory(point, frame, frame_0)

# Create a new particle of mass m[i] at this point
Pa0 = Particle('Pa0', point0, m[0])
bodies=Pa0

# Set forces & compute kinematic ODE
loads = point0, m[0] * g * frame_0.x
kd_eqs = q[0].diff(t) - u[0]


method = KanesMethod(frame, q_ind=q, u_ind=u, kd_eqs=[kd_eqs])
fr, fr_star = method.kanes_equations([bodies],[loads])


initial_positions = 3 * np.pi / 4
initial_velocities = 0
y0 = [initial_positions,initial_positions]
lengths = [.5]
masses = [1.]

par = [g, l,m]
par_val = [9.81,lengths,masses]

dummy_symbols = [Dummy() for i in q + u]  # Create a dummy symbol for each variable
dummy_dict = dict(zip(q + u, dummy_symbols))
print(dummy_dict)
kds = method.kindiffdict()

# substitute unknown symbols for qdot terms
mm_sym = method.mass_matrix_full.subs(kds).subs(dummy_dict)
fo_sym = method.forcing_full.subs(kds).subs(dummy_dict)
print(dummy_symbols + par)
# create functions for numerical calculation
mm_func = lambdify(dummy_symbols + par, mm_sym)
fo_func = lambdify(dummy_symbols + par, fo_sym)


def f(y, t, args):
    print(f'y {y} args {args}')
    vals = np.concatenate((y, args))
    print(vals)
    print(*vals)

    sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))

    return np.array(sol).T[0]

temp  = odeint(f, y0, time, args=(par_val,))

def get_xy_coords(p, lengths):
    """Get (x, y) coordinates from generalized coordinates p"""
    # p = np.atleast_2d(p)
    n = p.shape[1] // 2
    print(f'n {n}')
    zeros = np.zeros(p.shape[0])
    print(zeros.shape, zeros.shape[0])

    x  = np.hstack([zeros, lengths[0] * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths[0] * np.cos(p[:, :n])])


    return np.cumsum(x, 1), np.cumsum(y, 1)

x, y = get_xy_coords(temp,lengths)

plt.plot(x,time)
plt.plot(y,time)

plt.show()

