
import numpy as np
import prettytable

# arbitrarty demension
# How to account for the mass
#compute acell
import scipy.constants as constants
G =constants.G

def gravity(rij, m):
    radious = np.linalg.norm(rij)
    return G * m[0] *m[1] *rij/radious**3


def n_body(t,y,p):
    '''
    p={masses: { floats }}
    x = [x1,y1,x2,y2  ....,vx1,vy1,]
    Write what goes in here!
    Instructions above.
    '''

    dim = p['dimension']
    masses = p['m']
    N = len(masses)
    # force = p['force'] # force function
    half =dim //2
    accel = np.zeros((N, N, dim))

    print(y.shape)
    for i in range(0, y.size//2, dim):
        for j in range(i + dim, y.size//2, dim):
            rij = y[i:i+dim] - y[j:j+dim]
            f = gravity(rij, [masses[i//dim], masses[j//dim]])
            accel[i//dim][j//dim] = f / masses[j//dim]
            accel[j//dim][i//dim] = - f / masses[i//dim]


    dv_dt = np.sum(accel, axis=1).flatten().tolist()
    dy_dt = y[y.size//2:]
    print('dv,dy',dv_dt,dy_dt)
    dxdt =  np.array([dv_dt,dy_dt]).flatten()
    print(f'len {len(dxdt)}',dxdt)
    if p['fix_first']:
        dxdt[:dim] = 0
        dxdt[half: half+dim] = 0
    return dxdt

def runge_kutta4(dt, f, y, t, p):
    k1 = dt*f(t,y,p)
    k2 = dt*f(t+.5*dt,y+.5*k1,p)
    k3 = dt*f(t+.5*dt,y+.5*k2,p)
    k4 = dt * f(t+dt,y+k3,p)
    rk4 = (1./6.) * k1 + (2./6.)*k2 + (2./6.) * k3 + (1./6.)*k4
    return rk4

# def n_body_manager(time, bodies,force, fix_first):
#     dt = time[1]-time[0]
#     state_vectors = []
#     masses = []
#
#     for body in bodies:
#         state_vectors.append(body.r)
#         masses.append(body.T)
#     y = np.ravel(state_vectors.T)
#     p = {'bodies': len(bodies),
#          'mass' : masses,
#
#          'force':force,
#          'fix_first':fix_first,
#          'fixed':np.max(masses),
#          'dimensions': bodies[0].d,
#
#          }
#     print(f'y {y}')
#     t = time[0]
#     while t< time[-1]:
#         temp = runge_kutta4(dt,n_body,y,t,p)
#         y += temp
params = {'bodies': 3,
          'dimension': 2,
          'm': [10, 20, 30],
          'force': gravity,
          }

"""
Test data - make sure that your function reproduces the following.
Why not add a row of differences and make sure they are zero?
"""

# Order is all positions then all velocities grouped by body.
# eg, three bodies in two dimensions:
# x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3

from prettytable import PrettyTable


euler      = np.array([0,0,1,0,-1,0,0,0,0,.8,0,-.8])
four_body  = np.array([1.382857,0,\
                   0,0.157030,\
                  -1.382857,0,\
                   0,-0.157030,\
                   0,0.584873,\
                   1.871935,0,\
                   0,-0.584873,\
                  -1.871935,0],dtype=np.float128)
helium_1 = np.array([0,0,2,0,-1,0,0,0,0,.95,0,-1])

# The data structure holding parameters. You need not
# do it this way, but it's nice.

p = {'m':[1,1,1],'G':1,'dimension':2,'fix_first':False}
p4 = {'m':[1,1,1,1],'G':1,'dimension':2,'fix_first':False}
phe = {'m':[2,-1,-1],'G':1,'dimension':2,'fix_first':True}

headings = ['RUN','x1','y1','x2','y2','x3','y3','vx1','vy1','vx2','vy2','vx3','vy3']
t = PrettyTable(headings)
t.add_row(['euler']+list(n_body(0,euler,p)))

t.add_row(['He']+list(n_body(0,helium_1,phe)))
print(t)

headings = ['RUN','x1','y1','x2','y2','x3','y3','x4','y4','vx1','vy1','vx2','vy2','vx3','vy3','vx4','vy4']
t = PrettyTable(headings)
t.add_row(['4 body']+list(n_body(0,four_body,p4)))
print(t)
