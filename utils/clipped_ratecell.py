from jax import numpy as jnp, random, jit

from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.model_utils import create_function, threshold_soft, \
                                       threshold_cauchy
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2, step_rk4


def _dfz_internal_gaussian(z, j, j_td, tau_m, leak_gamma):
    z_leak = z
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_laplace(z, j, j_td, tau_m, leak_gamma):
    z_leak = jnp.sign(z)
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_cauchy(z, j, j_td, tau_m, leak_gamma):
    z_leak = (z * 2)/(1. + jnp.square(z))
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz_internal_exp(z, j, j_td, tau_m, leak_gamma):
    z_leak = jnp.exp(-jnp.square(z)) * z * 2
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _modulate(j, dfx_val):
    return j * dfx_val

def _run_cell(dt, j, j_td, z, tau_m, leak_gamma=0., integType=0, priorType=0):
    _dfz_fns = {
        0: lambda t, z, params: _dfz_internal_gaussian(z, *params),
        1: lambda t, z, params: _dfz_internal_laplace(z, *params),
        2: lambda t, z, params: _dfz_internal_cauchy(z, *params),
        3: lambda t, z, params: _dfz_internal_exp(z, *params),
    }
    _dfz_fn = _dfz_fns.get(priorType, _dfz_internal_gaussian)
    _step_fns = {
        0: step_euler,
        1: step_rk2,
        2: step_rk4,
    }
    _step_fn = _step_fns.get(integType, step_euler)
    params = (j, j_td, tau_m, leak_gamma)
    _, _z = _step_fn(0., z, _dfz_fn, dt, params)
    return _z

def _run_cell_stateless(j):
    return j + 0


class ClippedRateCell(JaxComponent):
    """
    A rate-coded cell with pre-activation clipping to prevent overflow.

    Same dynamics as the standard RateCell but adds jnp.clip(-5, 5) on
    the pre-activation state value to keep z bounded regardless of input
    pressure. Uses only j and j_td compartments (no attention-specific
    compartments).

    | tau_m * dz/dt = -leak_gamma * prior(z) + (j + j_td)
    | z = clip(z, -5, 5)
    | zF = f(z)

    | --- Cell Input Compartments: ---
    | j - bottom-up input pressure
    | j_td - top-down input pressure
    | --- Cell State Compartments ---
    | z - rate activity
    | --- Cell Output Compartments: ---
    | zF - post-activation function activity, i.e., fx(z)
    """

    def __init__(
            self, name, n_units, tau_m, prior=("gaussian", 0.), act_fx="identity", output_scale=1., threshold=("none", 0.),
            integration_type="euler", batch_size=1, resist_scale=1., shape=None, is_stateful=True, **kwargs):
        jax_comp_kwargs = {k: v for k, v in kwargs.items() if k not in ('omega_0',)}
        this_class_kwargs = {k: v for k, v in kwargs.items() if k in ('omega_0',)}
        super().__init__(name, **jax_comp_kwargs)

        self.output_scale = output_scale
        self.tau_m = tau_m
        self.is_stateful = is_stateful
        if isinstance(tau_m, float):
            if tau_m <= 0:
                self.is_stateful = False
        priorType, leakRate = prior
        priorTypeDict = {
            "gaussian": 0,
            "laplacian": 1,
            "cauchy": 2,
            "exp": 3
        }
        self.priorType = priorTypeDict.get(priorType, 0)
        self.priorLeakRate = leakRate
        thresholdType, thr_lmbda = threshold

        self.thresholdType = thresholdType
        self.thr_lmbda = thr_lmbda
        self.resist_scale = resist_scale

        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        _shape = (batch_size, n_units)
        if shape is None:
            shape = (n_units,)
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size

        omega_0 = None
        if act_fx == "sine":
            omega_0 = this_class_kwargs["omega_0"]
        self.fx, self.dfx = create_function(fun_name=act_fx, args=omega_0)

        restVals = jnp.zeros(_shape)
        self.j = Compartment(restVals, display_name="Input Stimulus Current", units="mA")
        self.zF = Compartment(restVals, display_name="Transformed Rate Activity")
        self.j_td = Compartment(restVals, display_name="Modulatory Stimulus Current", units="mA")
        self.z = Compartment(restVals, display_name="Rate Activity", units="mA")

    @compilable
    def advance_state(self, dt):
        j = self.j.get()
        j_td = self.j_td.get()
        z = self.z.get()

        if self.is_stateful:
            dfx_val = self.dfx(z)
            j = _modulate(j, dfx_val) * self.resist_scale
            tmp_z = _run_cell(
                dt, j, j_td, z, self.tau_m, leak_gamma=self.priorLeakRate, integType=self.intgFlag,
                priorType=self.priorType
            )
            if self.thresholdType == "soft_threshold":
                tmp_z = threshold_soft(tmp_z, self.thr_lmbda)
            elif self.thresholdType == "cauchy_threshold":
                tmp_z = threshold_cauchy(tmp_z, self.thr_lmbda)
            z = jnp.clip(tmp_z, -3.0, 3.0)  ## pre-activation clipping
            zF = self.fx(z) * self.output_scale
        else:
            j_total = j + j_td
            z = _run_cell_stateless(j_total)
            zF = self.fx(z) * self.output_scale

        self.j.set(j)
        self.j_td.set(j_td)
        self.z.set(z)
        self.zF.set(zF)

    @compilable
    def reset(self):
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape)

        self.j.set(restVals)
        self.j_td.set(restVals)
        self.z.set(restVals)
        self.zF.set(restVals)
