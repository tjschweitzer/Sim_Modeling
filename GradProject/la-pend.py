from __future__ import division
import sympy as sy

from sympy.physics.mechanics import *
q1 = dynamicsymbols('q1')                     # Angle of pendulum
u1 = dynamicsymbols('u1')                     # Angular velocity
q1d = dynamicsymbols('q1', 1)
L, m, t, g = sy.symbols('L, m, t, g')
N = ReferenceFrame('N')
pN = Point('N*')
pN.set_vel(N, 0)




# Redefine A and P in terms of q1d, not u1
A = N.orientnew('A', 'axis', [q1, N.z])
A.set_ang_vel(N, q1d*N.z)
P = pN.locatenew('P', L*A.x)
vel_P = P.v2pt_theory(pN, N, A)
pP = Particle('pP', P, m)

kde = sy.Matrix([q1d - u1])

# Input the force resultant at P
R = m*g*N.x

A = N.orientnew('A', 'axis', [q1, N.z])
A.set_ang_vel(N, q1d*N.z)
P = pN.locatenew('P', L*A.x)
vel_P = P.v2pt_theory(pN, N, A)
pP = Particle('pP', P, m)

# Solve for eom with Lagrange's method
Lag = Lagrangian(N, pP)
LM = LagrangesMethod(Lag, [q1], forcelist=[(P, R)], frame=N)
lag_eqs = LM.form_lagranges_equations()
print(lag_eqs)