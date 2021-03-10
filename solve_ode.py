
def Euler(f,dt,t,y,args):
    """
    given y(t)
    :return: y(t+delta_t)
    """
    return y + f(t,y,*args)* dt

def EulerCromer(f,dt,t,y,args):
    y_end = Euler(f,dt,t,y,args)
    return y + f(t+dt,y_end,*args) * dt


def EulerRichardson(f,dt,t,y,args):
    y_mid = Euler(f,dt,t+dt/2,y,args)
    return y + f(t+dt/2,y_mid,*args) * dt


def solve_ode(f,tspan, y0, method = Euler, *args, **options):
    t_0 = tspan[0]
    t_f = tspan[1]
    d_t = options['first_step']
    t= [t_0]
    y = [y0]

    while t[-1]<= t_f:
        y.append(method(f,d_t,t[-1],y[-1],args))
        t.append(t[-1]+d_t)
    return np.array(t),np.array(y)