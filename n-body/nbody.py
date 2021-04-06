import time
import numpy as np

def function_counter(fn):
    def wrapper(*args ):
        wrapper.called += 1
        return fn(*args)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper

def gravity (G, m1, m2, r):
    """ Computes the gravitational force between two bodies with large mass """
    return (-G * m1 * m2 / np.power(np.linalg.norm(r), 3)) * r

@function_counter
def n_body(t, y, p):
   # Get some constants and initialize\n",
    N = len(p['m'])
    d = p['dimension']
    Fmatrix = np.zeros([N, N, d])

    # split y into position and velocity vectors, go from flattened pos. vector to size Nxd array\n",
    half = y.size // 2
    pos_vec = y[:half]
    vel_vec = y[half:]
    pos_matrix = pos_vec.reshape(N, d)

    # Loop over the top right corner of the force matrix\n",
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            # Find the vector from body j to body i\n",
            rij = pos_matrix[i, :] - pos_matrix[j, :]

            # Compute the force from body j onto body i\n",
            Fij = p['force'](p['G'], p['m'][i], p['m'][j], rij)

            # Fill in the symmetric pieces of the force matrix\n",
            Fmatrix[i, j, :] = Fij / p['m'][i]
            Fmatrix[j, i, :] = -Fij / p['m'][j]

    # Compute the force vectors acting on each body\n",
    forces = np.sum(Fmatrix, axis=1)

    # flatten the forces matrix and combine it with the vector list\n",
    # TODO: Rework how you build these\n",
    acc_vec = forces.flatten().tolist()
    dxdt = np.array([vel_vec, acc_vec]).flatten()

    if p['fix_first']:
        # Find indices that need to be zero\n",
        dxdt[:d] = 0
        dxdt[half:half + d] = 0

    return dxdt

def runge_kutta4( f,t,y,dt, p,options):
    k1 = f(t,y,*p)*dt
    k2 = f(t+.5*dt,y+.5*k1,*p)*dt
    k3 = f(t+.5*dt,y+.5*k2,*p)*dt
    k4 =  f(t+dt,y+k3,*p)*dt
    rk4 =y+ (k1 + 2 * k2 + 2 * k3 + k4)/6.
    return rk4,dt
