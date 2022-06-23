import numpy as np

def RK4(f, x0, v0, w, ts, *params):
    n = len(ts)
    dt = ts[1]-ts[0]
    xs, vs = np.zeros(n), np.zeros(n)
    x, v = x0, v0
    for i, t in enumerate(ts):
        xs[i], vs[i] = x, v
        k0 = dt*f(t, x, v, w, *params)
        l0 = dt*v
        k1 = dt*f(t+0.5*dt, x+0.5*l0, v+0.5*k0, w, *params)
        l1 = dt*(v+0.5*k0)
        k2 = dt*f(t+0.5*dt, x+0.5*l1, v+0.5*k1, w, *params)
        l2 = dt*(v+0.5*k1)
        k3 = dt*f(t+dt, x+l2, v+k2, w, *params)
        l3 = dt*(v+k2)
        v += (k0 + 2*k1 + 2*k2 + k3)/6.
        x += (l0 + 2*l1 + 2*l2 + l3)/6.
    return xs, vs