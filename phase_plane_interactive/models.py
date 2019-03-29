class G2DO:
    "Generic nonlinear 2-D (phase plane) oscillator."
    state = 'W', 'V'
    limit = (-5, 5), (-5, 5)
    input = 'c_0'
    param = 'a'
    const = {'tau': 1.0, 'I': 0.0, 'a': -2.0, 'b': -10.0, 'c': 0.0, 'd': 0.02,
             'e': 3.0, 'f': 1.0, 'g': 0.0, 'alpha': 1.0, 'beta': 1.0,
             'gamma': 1.0}
    drift = (
        'd * tau * (alpha*W - f*V**3 + e*V**2 + g*V + gamma*I + gamma*c_0)',
        'd * (a + b*V + c*V**2 - beta*W) / tau'
    )
    diffs = 1e-3, 1e-3
    obsrv = 'W', 'V'
