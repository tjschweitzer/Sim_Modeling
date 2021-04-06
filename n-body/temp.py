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

    return np.array(t), np.array(y), option_list, dt_list
