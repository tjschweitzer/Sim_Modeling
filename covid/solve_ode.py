#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.interpolate import interp1d

def Euler(f, t, y, dt, args, options):
    return y + dt * f(t, y, *args), dt


def EulerCromer(f, t, y, dt, args, options):
    yn = y + dt * f(t, y, *args)
    tn = t + dt
    return y + dt * f(tn, yn, *args, options), dt


def EulerRichardson(f, t, y, dt, args, options):
    ymid = y + 0.5 * dt * f(t, y, *args)
    tmid = t + 0.5 * dt
    return y + dt * f(tmid, ymid, *args), dt


def solve_ode(f, tspan, y0, method=Euler, *args, **options):
    t_0 = tspan[0]
    t_f = tspan[1]
    d_t = options["first_step"]
    y = []
    t = []
    t.append(tspan[0])
    y.append(y0)
    dt = options["first_step"]
    option_list = [options]
    dt_list = [dt]
    if method in (Euler, EulerCromer, EulerRichardson, runge_kutta4):

        while t[-1] < tspan[1]:
            y_i, dt = method(f, t[-1], y[-1], dt, args, options)
            y.append(y_i)
            t.append(t[-1] + dt)
            option_list.append(options)
            dt_list.append(dt)
    elif method is RK45:
        options["FSAL"] = False
        options["last_k"] = 0
        while t[-1] < tspan[1]:
            current_t = t[-1]
            new_y, dt, options["FSAL"], options["last_k"] = method(
                y[-1], dt, f, t[-1], *args, **options
            )
            y.append(new_y)
            t.append(t[-1] + dt)
            dt_list.append(dt)
            
        f = interp1d(np.array(t), np.array(y), axis=0)
        t = np.arange(t_0, t_f, d_t)
        y = f(t)

    return np.array(t), np.array(y), option_list, dt_list

def RK45(y, dt, f, t, *p, **options):

    S = 0.9998
    b = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    b_star = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    )
    c = np.array([0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1, 1])
    a = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ]
    )
    FSAL = options["FSAL"]
    last_k = options["last_k"]
    atol = options["atol"]
    rtol = options["rtol"]
    if FSAL:
        k1 = last_k
    else:
        k1 = dt * f(t, y, *p)
    y_2 = y + a[1, 0] * k1
    k2 = dt * f(t + c[0] * dt, y_2, *p)
    y_3 = y + a[2, 0] * k1 + a[2, 1] * k2
    k3 = dt * f(t + c[1] * dt, y_3, *p)
    y_4 = y + a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3
    k4 = dt * f(t + c[2] * dt, y_4, *p)
    y_5 = y + a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4
    k5 = dt * f(t + c[3] * dt, y_5, *p)
    y_6 = y + a[5, 0] * k1 + a[5, 1] * k2 + a[5, 2] * k3 + a[5, 3] * k4 + a[5, 4] * k5
    k6 = dt * f(t + dt, y_6, *p)
    y_7 = (
        y
        + a[6, 0] * k1
        + a[6, 1] * k2
        + a[6, 2] * k3
        + a[6, 3] * k4
        + a[6, 4] * k5
        + a[6, 5] * k6
    )
    k7 = dt * f(t + dt, y_7, *p)

    k = np.row_stack([k1, k2, k3, k4, k5, k6, k7])
    y_new = y + np.dot(b, k)

    delta = np.dot(b - b_star, k)
    scale = atol + rtol * np.abs(y_new)
    error = np.sqrt(np.mean((delta / scale) ** 2))

    dt_new = S * dt * np.power(1 / error, 1 / 5)

    if error > 1:
        options["FSAL"] = False
        options["last_k"] = 0
        dt_new = max(dt_new, dt / 5)
        y_new, dt_new, FSAL, last_k = RK45(y, dt_new, f, t, *p, **options)
    else:
        dt_new = min(dt_new, dt * 10)
        FSAL = True
        last_k = k7

    return y_new, dt_new, FSAL, last_k

