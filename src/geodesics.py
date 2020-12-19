#!/usr/bin/env python3
"""Compute geodesics given a metric

We will just numerically integrate the geodesic equation. For the Kerr metric,
Levin Perez-Giz 2008 (A Periodic Table of Black Hole Orbits) uses a Hamiltonian
but they verify that they get the same result as one gets from the geodesic
equation (though presumably the former is more accurate or faster?).

In n dimensions, our metrics must either
1. Return an n x n matrix g for the metric itself, along
 with derivs[:,:,:] where derivs[a,:,:] is the partial derivative of the entries of
 g with respect to variable a.
2. As above, but only the diagonals are used (and off-diagonals are assumed to be zero).
 See examples.

In all metrics that involve speed of light c, we assume c=1.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate
import scipy.linalg as SLA

def schwarzschild_metric(x, r_s):
    """Return Schwarzschild metric (and its derivs) at a given point, for
    Schwarzschild radius r_s.

    Coordinates are (t, r, theta, phi) where theta is angle from north pole
    and phi is "east longitude".  (These are Schwarzschild coordinates.)

    We let c=1.  If we didn't, the dt^2 term would also have a factor of c^2
    in it.

    g = -(1 - r_s/r) dt^2 + (1 + r_s/r)^(-1) dr^2 + r^2 g_Omega
    where g_Omega = dtheta^2 + sin^2 theta dphi^2.  What's nice about this
    is that in these coordinates, g is always diagonal.

    Returns:
        (g, derivs) where g has shape (4,) (the diagonal of the matrix g_ij)
        and derivs[i,:] is the diagonal of the partial derivative of (that diagonal)
        with respect to variable i.
    """
    g = np.zeros(4)
    r = x[1]
    r2 = r * r
    theta = x[2]
    one_minus_r_s_over_r = 1 - r_s/r
    g[0] = -one_minus_r_s_over_r # -(1 - r_s/r) dt^2
    g[1] = 1 / one_minus_r_s_over_r
    g[2] = r2
    sin2_theta = np.sin(theta) ** 2
    g[3] = r2 * sin2_theta
    derivs = np.zeros((4,4))
    # Note that d(1 - r_s/r) / dr = r_s r^(-2) =: div_term
    div_term = r_s / r2
    # derivs[0,:] is all 0, because no dependence on t above
    # derivs[3,:] is all 0, because no dependence on phi above
    # derivs with respect to r:
    derivs[1,0] = -div_term
    derivs[1,1] = -div_term / (one_minus_r_s_over_r ** 2)
    derivs[1,2] = 2 * r
    derivs[1,3] = 2 * r * sin2_theta
    # derivs with respect to theta:
    # d(sin^2(theta))/dtheta = 2sin(theta)cos(theta) = sin(2*theta)
    # only the r^2 sin^2(theta) dphi^2 term depends on theta
    derivs[2,3] = r2 * np.sin(2 * theta)
    return g, derivs

def schwarzschild_metric_alt(x, r_s):
    """Like above, but don't take advantage of fact that g is diagonal.  For testing
    our code for the general case."""
    g, derivs = schwarzschild_metric(x, r_s)
    gfull = np.diag(g)
    derivsfull = np.zeros((4,4,4))
    for k in range(4):
        derivsfull[k,:,:] = np.diag(derivs[k,:])
    return gfull, derivsfull

def kerr_metric(x, r_s, a):
    """
    Kerr metric.  Conventions as above, though now we also have a := J / (M c) where J = angular momentum,
    M = mass of the object, r_s = 2 G M / c^2 (though we assume c = 1).
    (presumably positive J means CCW about z axis if looking down "from above" in right-handed coordinates?)

    Coordinates are (t, r, theta, phi) but they are Boyer-Lindquist coordinates.  In particular, if you want
    to convert to Cartesian, you have:
    x = sqrt(r^2 + a^2) sin theta cos phi
    y = sqrt(r^2 + a^2) sin theta sin phi
    z = r cos theta

    We let Sigma = r^2 + a^2 cos^2 theta
    Delta = r^2 - r_s r + a^2

    Note that the Kerr metric is not quite diagonal: there is a dt dphi cross-term.  In particular, return
    values are not just diagonals.

    """
    g = np.zeros((4,4))
    derivs = np.zeros((4,4,4))
    r = x[1]
    theta = x[2]
    #costheta = np.cos(theta)
    sintheta = np.sin(theta)
    sin2theta = sintheta * sintheta
    cos2theta = 1 - sin2theta
    a2 = a * a
    r2 = r * r
    r2pa2 = r2 + a2
    Sigma = r2 + a2 * cos2theta
    Delta = r2pa2 - r_s * r
    rsr_over_Sigma = r_s * r / Sigma
    # Note that off-diagonal entries contribute a dt dphi and dphi dt term, so the coeffs
    # there are half the coeff of dt dphi you see in written formulas.
    g[0,0] = -(1 - rsr_over_Sigma)
    g[1,1] = Sigma / Delta
    g[2,2] = Sigma
    g[3,3] = (r2pa2 + rsr_over_Sigma * a2 * sin2theta) * sin2theta
    g[0,3] = -(rsr_over_Sigma * a * sin2theta)
    g[3,0] = g[0,3]

    # Now, the partial derivatives.
    # There is no dependence on t or phi, so those derivs are 0.
    # d/dr:
    Sigmap = 2 * r
    Deltap = 2 * r - r_s
    rsr_over_Sigmap = r_s * (Sigma - r * Sigmap) / (Sigma * Sigma)
    derivs[1,0,0] = rsr_over_Sigmap
    derivs[1,1,1] = (Delta * Sigmap - Sigma * Deltap) / (Delta * Delta)
    derivs[1,2,2] = Sigmap
    derivs[1,3,3] = (2 * r + rsr_over_Sigmap * a2 * sin2theta) * sin2theta
    derivs[1,0,3] = -(rsr_over_Sigmap * a * sin2theta)
    derivs[1,3,0] = derivs[1,0,3]

    # d/dtheta
    # (sin^2 theta)' = 2 sin theta cos theta = sin(2 theta)
    # (cos^2 theta)' = -2 sin theta cos theta = - sin(2 theta)
    sin2thetap = np.sin(2 * theta) # (sin^2 theta)'
    cos2thetap = -sin2thetap
    Sigmap = a2 * cos2thetap
    # Deltap would be 0
    # (r_s r / Sigma)' = - r_s r / Sigma^2 * Sigma'
    rsr_over_Sigmap = -rsr_over_Sigma * Sigmap / Sigma
    derivs[2,0,0] = rsr_over_Sigmap
    derivs[2,1,1] = Sigmap / Delta
    derivs[2,2,2] = Sigmap
    derivs[2,3,3] = ( a2 * (rsr_over_Sigmap * sin2theta + rsr_over_Sigma * sin2thetap) * sin2theta +
                      (r2pa2 + rsr_over_Sigma * a2 * sin2theta) * sin2thetap )
    derivs[2,0,3] = -a * (rsr_over_Sigmap  * sin2theta + rsr_over_Sigma * sin2thetap)
    derivs[2,3,0] = derivs[2,0,3]

    return g, derivs

def poincare_upper_half_plane(x_):
    """
    Poincare metric tensor for upper half plane:
    ds^2 = (dx^2 + dy^2) / y^2

    That's where x = x_[0], y = x_[1].  (Underscore is just to distinguish the vector x_ from its
    first coordinate which we're also calling x.)

    Diagonal like Schwarzschild
    """
    y = x_[1]
    y2 = y * y
    g = np.ones(2) / y2
    derivs = np.zeros((2,2))
    # derivs[0,:] = 0 because no reliance on x.
    # d(1/y^2)/dy = -2 y ^ (-3)
    derivs[1,:] = -2/(y * y2)
    return g, derivs

def poincare_upper_half_plane_alt(x_):
    """Full-matrix version of poincare_upper_half_plane.  For testing."""
    g, derivs = poincare_upper_half_plane(x_)
    gfull = np.diag(g)
    derivsfull = np.zeros((2,2,2))
    for k in range(2):
        derivsfull[k,:,:] = np.diag(derivs[k,:])
    return gfull, derivsfull

# Note that in some of the code below, there are formulas that might not be correct
# when g is not symmetric (but of course, the metric is always symmetric).
def apply_christoffel_diagonal(g, derivs, u, v):
    """
    Given metric g and its partial derivs w.r.t. each variable, compute
    Gamma^a_bc u^b v^c where Gamma^a_bc are the Christoffel symbols.

    This is specialized to the special case where g and its derivs are diagonal (so variable g is diagonal
    of the metric g_ij, and derivs[i,:] is diagonal of deriv of metric w.r.t. variable i).

    Now, we have
       Gamma^a_bc = 0.5 g^ad (g_cd,b + g_bd,c - g_bc,d)
    where notation g_cd,b means partial deriv of g_cd with respect to variable b.
    Because of our diagonal assumption, the index raising will be trivial.  So, first we introduce
    Kappa_dbc = (g_cd,b + g_bd,c - g_bc,d)
    """
    (n,) = g.shape # dimensionality
    # If we contract g_cd,b with u^b, we get a linear combination of the entries of [derivs],
    # which would be obtained as u.dot(derivs).
    # The result is the diagonal of g_cd; contracting that with v^c is pointwise product of that
    # diagonal with v^c
    g_cd_comma_b_term = u.dot(derivs) * v
    # g_bd,c is similar, but role of u and v is reversed.
    g_bd_comma_c_term = v.dot(derivs) * u
    # The g_bc,d term is a little more awkward to think about.  It's ith entry is derivative
    # of g_bc with respect to ith variable.  That g_bc would be diagonal.  When contracted
    # with u^b and v^c, that's like taking pointwise product of u, v, and that diagonal, and
    # then summing.  The b index gets reused...
    g_bc_comma_d_term = np.einsum("db,b,b->d", derivs, u, v)
    # That should be equivalent to the following.
    tmp = (derivs * u).dot(v)
    assert np.allclose(tmp, g_bc_comma_d_term)
    Kappa = g_cd_comma_b_term + g_bd_comma_c_term - g_bc_comma_d_term
    # Now, have to raise index and multiply by 0.5.  Since g is diagonal, raising an index
    # is just pointwise division by (diagonal of) g.
    return Kappa / (2 * g)

def apply_christoffel_general(g, derivs, u, v):
    """As above, but no assumption of diagonality; g is a matrix, and derivs[i,:,:]
    is its partial derivative w.r.t. variable i.

    Note that roundoff error can make this come out a little deifferent even when g
    is diagonal.
    """
    (n, _n) = g.shape
    # Note that g_cd = g_dc
    g_cd_comma_b_term = np.einsum('b,bcd,c->d', u, derivs, v) #u.dot(derivs).dot(v)
    g_bd_comma_c_term = np.einsum('c,cbd,b->d', v, derivs, u) #v.dot(derivs).dot(u)
    g_bc_comma_d_term = np.einsum("dbc,b,c->d", derivs, u, v)
    Kappa = g_cd_comma_b_term + g_bd_comma_c_term - g_bc_comma_d_term
    # Now, have to raise index and multiply by 0.5.  Since g is diagonal, raising an index
    # is just pointwise division by (diagonal of) g.
    # return 0.5 * SLA.solve(g, Kappa, assume_a='sym')
    # We'll actually do without assuming symmetric; for these small systems I don't think
    # there's much advantage, and in the special case where g is diagonal, we get bitwise
    # identical results using the general solver but epsilon different results when assuming
    # sym.
    return 0.5 * SLA.solve(g, Kappa)

def apply_christoffel(g, derivs, u, v):
    """Like above, but detects whether we have diagonal of g or all of g"""
    gdims = len(g.shape)
    if gdims == 1:
        return apply_christoffel_diagonal(g, derivs, u, v)
    elif gdims == 2:
        return apply_christoffel_general(g, derivs, u, v)
    else:
        assert False, f"bad gdims {gdims}"

def solve(metric, extra_params, x0, xdot0, max_s, max_step_size = None):
    """
    Get trajectory.

    xdot denotes derivative with respect to s, which is time in the simulation and not necessarily
    related to any kind of time coordinate that your space itself might have.
    """
    (n,) = x0.shape
    y0 = np.concatenate([x0, xdot0])
    #print(f"y0={y0}")
    def func(_t, y):
        (two_n,) = y.shape
        assert two_n == 2 * n
        x = y[:n]
        xdot = y[n:]
        g, derivs = metric(x, *extra_params)
        #xdotdot = -apply_christoffel_diagonal(g, derivs, xdot, xdot)
        xdotdot = -apply_christoffel(g, derivs, xdot, xdot)
        ydot = np.concatenate([xdot, xdotdot])
        return ydot
    if max_step_size is None:
        t_eval = None
    else:
        # have to also call int b/c np.float64 rounds to np.float64
        # Strictly speaking, this could yield steps slightly larger than
        # max step size if the following rounds down.
        t_eval = np.linspace(0, max_s, int(round(max_s / max_step_size)))
    result = scipy.integrate.solve_ivp(func, (0, max_s), y0, t_eval=t_eval)
    if not result.success:
        print(f"Warning: solver failed ({result.message})")
        # Maybe result.y is still the part it got so far?
    # assert result.success, f"solver failed: {result.message}"
    # result.y uses index 1 the way are using index 0 elsewhere, so we transpose.
    return result.y.T, result.t

def eval_metric(metric, extra_params, x, xdot):
    g, _derivs = metric(x, *extra_params)
    if len(g.shape) == 1:
        return xdot.dot(g * xdot)
    else:
        return xdot.dot(g).dot(xdot)

def test_apply_christoffel():
    r_s = 1
    metric = schwarzschild_metric
    metric_ = schwarzschild_metric_alt # for testing
    extra_params = [r_s]

    r, phidot, max_s = 6, 0.045, 1000 # stable, stays between r=6 and r=4 ish
    x = np.array([0., r, np.pi/2, 0.])
    xdot = np.array([1., 0., 0., phidot])
    u = np.array([1., 0.2, 0.3, 0.5])
    v = np.array([1., 0.7, -0.1, 0.4])

    g, derivs = metric(x, *extra_params)
    g_, derivs_ = metric_(x, *extra_params)
    #res = apply_christoffel(g, derivs, xdot, xdot)
    #res_ = apply_christoffel(g_, derivs_, xdot, xdot)
    res = apply_christoffel(g, derivs, u, v)
    res_ = apply_christoffel(g_, derivs_, u, v)

def test_apply_christoffel_poincare():
    metric = poincare_upper_half_plane
    metric_ = poincare_upper_half_plane_alt
    extra_params = []

    x = np.array([0., 1.])
    u = np.array([1., 0.2])
    v = np.array([0.4, 0.7])

    g, derivs = metric(x, *extra_params)
    g_, derivs_ = metric_(x, *extra_params)
    #res = apply_christoffel(g, derivs, xdot, xdot)
    #res_ = apply_christoffel(g_, derivs_, xdot, xdot)
    res = apply_christoffel(g, derivs, u, v)
    res_ = apply_christoffel(g_, derivs_, u, v)

def test_metric(metric, extra_params, x, epsilon=1e-5):
    """
    Numerically differentiate a metric at a point to see how it compares to the `derivs`
    values it returns.
    """
    (n,) = x.shape
    g, derivs = metric(x, *extra_params)
    direction = np.random.randn(n)
    def g_(delta):
        return metric(x + delta * direction, *extra_params)[0]
    num_directional_deriv = (-g_(2*epsilon) + 8 * g_(epsilon)
                             - 8 * g_(-epsilon) + g_(-2*epsilon))/(12 * epsilon)
    directional_deriv = np.einsum('d,dij->ij', direction, derivs)
    print(directional_deriv)
    print(num_directional_deriv)
    print(num_directional_deriv - directional_deriv)

def test_some_metrics():
    metric = poincare_upper_half_plane_alt
    extra_params = []
    x = np.array([0.3, 0.44])
    test_metric(metric, extra_params, x)

    metric = kerr_metric
    r_s = 0.9
    a = 0.3 # J / (M c)
    extra_params = [r_s, a]
    x = np.array([0.3, 1.44, 0.7, 0.81])
    test_metric(metric, extra_params, x)




def test0():
    metric = poincare_upper_half_plane
    extra_params = []
    x0 = np.array([0., 1.])
    xdot0 = np.array([0.5, 0.5])
    max_s = 10
    max_step_size = 0.01
    # max_step_size = None
    x_xdot, s_values = solve(metric, extra_params, x0, xdot0, max_s,
                                  max_step_size=max_step_size)
    u = x_xdot[:,0]
    v = x_xdot[:,1]
    #plt.plot(u, v, '-o')
    plt.plot(u, v, '.', markersize=1)
    plt.show()

# Note: For solutions that get "sucked in", the solution stops at event horizon
# r=r_s (actually, just slightly before for numerical reasons); this isn't that
# it reached a singularity, it just reached a /coordinate singularity/.  A
# different coordinate system would let us look at what happens after it passes
# the event horizon.
#
# At r = 2 r_s, there's a subluminal velocity that gives us a circular orbit,
# but it's unstable: any slower and we get sucked in, any faster and we escape.
# (We need r > 3 r_s to get a stable circular orbit.)
# r = 1.5 r_s is where light can be in a circular orbit.
# Using tdot = 1, what value of phidot makes xdot null?
# 1 - r_s/r = 1/3, so we have a -1/3 dt^2.
# So we need 1.5^2 phidot^2 = 1/3
# phidot = np.sqrt(1/3 / 1.5 ** 2) = np.sqrt(4/27)
def test1():
    r_s = 1
    #metric = schwarzschild_metric
    metric = schwarzschild_metric_alt # for testing
    extra_params = [r_s]
    # coordinates are (t, r, theta, phi)
    # One thing that's a little annoying with this parametrization is that we
    # would need to set theta = pi/2 for equatorial motion (which is basically
    # what you want for Schwarzwild, given that you have spherical symmetry)
    # but np.cos(pi/2) doesn't evaluate exactly to 0.
    # If this is an issue, we could always specialize the metric to the case
    # where theta = pi/2, or change the parametrization so that theta = 0
    # is on the equator.  I'll just deal with roundoff error for now.
    max_s = 100
    r = 2.
    # phidot = 1. # escapes; motion is space-like
    # phidot = 0.5 # still escapes; still space-like
    # phidot = 0.05 # gets sucked in!
    # phidot = 0.25 # perfect circle!
    # phidot = 0.24 # gets sucked in
    phidot = 0.3 # escapes (but at least is time-like)
    phidot = 0.26 # escapes
    phidot = 0.2501 # escapes after a nearly circular orbit
    phidot = 0.2499 # sucked in after nearly circular orbit
    r, phidot = 1.5, np.sqrt(4/27) # circular orbit of light
    r, phidot = 6, 0.1 # escapes
    r, phidot = 6, 0.01 # STFI (sucked the  in)
    r, phidot, max_s = 6, 0.05, 1000 # stable, stays between r=6 and r=8 ish
    r, phidot, max_s = 6, 0.04, 1000 # sucked in
    r, phidot, max_s = 6, 0.045, 1000 # stable, stays between r=6 and r=4 ish
    x0 = np.array([0., r, np.pi/2, 0.])
    xdot0 = np.array([1., 0., 0., phidot])
    print(f"init ds^2 (negative is timelike) = {eval_metric(metric, extra_params, x0, xdot0)}")
    max_step_size = 0.1
    #max_step_size = None
    x_xdot, s_values = solve(metric, extra_params, x0, xdot0, max_s,
                                  max_step_size=max_step_size)
    t = x_xdot[:,0]
    r = x_xdot[:,1]
    theta = x_xdot[:,2]
    phi = x_xdot[:,3]
    assert np.allclose(theta, np.pi / 2)
    print(f"Last r = {r[-1]}")

    plt.polar(phi, r)
    plt.show()

def param_plot3d(x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x,y,z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def get_4momentum(metric, extra_params, x_xdot, mass):
    (two_n,) = x_xdot.shape
    n = two_n // 2
    x = x_xdot[:n]
    xdot = x_xdot[n:]
    g, _derivs = metric(x, *extra_params)
    # normalize w.r.t. tau to get 4 velocity u^alpha
    # then multiply by mass to get 4 momentum p^alpha (with index raised)
    # then lower index using metric
    dtau = np.sqrt(-(xdot.dot(g).dot(xdot)))
    u = xdot / dtau
    assert np.isclose(u.dot(g).dot(u), -1)
    momentum_vec = mass * u
    momentum_covec = g.dot(momentum_vec)
    return momentum_covec

# Kerr:
# Examples below could be bogus; they were based on a buggy implementation
# Strange example 1:
# Starts spinning in direction of black hole as it approaches, but gets sucked in

# Example 2:
# orbits several times (with radius going between 2ish and 8ish) and eventually gets sucked
# in
# Example 3:
# Starts out orbiting CW, slows down, gets pulled in, starts orbiting CCW and gets sucked in
# Example 4: phidot 0 at first, but some (negative) thetadot.
# Viewed from above, it starts rotating *CW* a little at first, then gets sucked in (and
# is starting to orbit CCW as this happens).
def test_kerr():
    r_s = 1
    a = 0.3 # J / (M c)
    metric = kerr_metric
    extra_params = [r_s, a]
    r, phidot, thetadot, max_s = 2., 0.3, 0, 100 # escapes
    r, phidot, thetadot, max_s = 2., -0.3, 0, 100 # escapes, but after more angle goes by, but less time?
    r, phidot, thetadot, max_s = 2., 0, 0, 7 # Example 1 (see above)
    r, phidot, thetadot, max_s = 2., 0.25, 0, 10000 # Example 2
    r, phidot, thetadot, max_s = 2., -0.25, 0, 1000 # Example 3
    r, phidot, thetadot, max_s = 2., 0, -0.25, 20 # Example 4
    r, phidot, thetadot, max_s = 2., 0, -0.3, 20 # Example 4

    x0 = np.array([0., r, np.pi/2, 0.])
    xdot0 = np.array([1., 0., thetadot, phidot])
    print(f"init ds^2 (negative is timelike) = {eval_metric(metric, extra_params, x0, xdot0)}")

    max_step_size = 0.1
    #max_step_size = None
    x_xdot, s_values = solve(metric, extra_params, x0, xdot0, max_s,
                                  max_step_size=max_step_size)
    t = x_xdot[:,0]
    r = x_xdot[:,1]
    theta = x_xdot[:,2]
    phi = x_xdot[:,3]
    assert thetadot != 0 or np.allclose(theta, np.pi / 2)
    print(f"Last r = {r[-1]}")

    # The variable names in Boyer-Lindquist coordinates are a little confusing because
    # the thing they call r doesn't yield (x,y) in the way you'd expect.
    r_xy = np.sqrt(r * r + a * a)
    x = r_xy * np.sin(theta) * np.cos(phi)
    y = r_xy * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    param_plot3d(x,y,z)

    plt.polar(phi, r_xy * np.sin(theta))
    plt.show()

    plt.polar(phi, r)
    plt.show()

    plt.plot(t, x)
    plt.plot(t, y)
    plt.plot(t, z)
    plt.legend(['x', 'y', 'z'])
    plt.show()

    mass = 1
    momenta = np.array([get_4momentum(metric, extra_params, x_xdot[i,:], mass)
                        for i in range(x_xdot.shape[0])])
    plt.plot(momenta)
    plt.legend(['rho_t', 'rho_r', 'rho_theta', 'rho_phi'])
    plt.grid()
    plt.show()
    # rho_t and rho_phi should be invariants


if __name__ == "__main__":
    test1()

