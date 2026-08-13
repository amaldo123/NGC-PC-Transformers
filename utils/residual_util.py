import jax.numpy as jnp
from jax import jit
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable

@jit
def add_residual(x1, x2):
    """Computes residual addition x1 + x2."""
    return x1 + x2

class ResidualComponent(JaxComponent):
    """
    ngclearn-compatible Residual Addition component.
    Receives shortcut x1 and transformed feature x2, outputs x1 + x2.
    Routes reverse error signal dmu equally to dx1 and dx2.
    """

    def __init__(self, name, shape, **kwargs):
        super().__init__(name, **kwargs)
        self.shape = shape

        # Forward compartments
        self.x1 = Compartment(jnp.zeros(shape))
        self.x2 = Compartment(jnp.zeros(shape))
        self.outputs = Compartment(jnp.zeros(shape))

        # Reverse / Error propagation compartments
        self.dmu = Compartment(jnp.zeros(shape))
        self.dx1 = Compartment(jnp.zeros(shape))
        self.dx2 = Compartment(jnp.zeros(shape))

    @compilable
    def advance_state(self):
        x1 = self.x1.get()
        x2 = self.x2.get()
        out = add_residual(x1, x2)
        self.outputs.set(out)

        # Reverse pass for error propagation (dmu -> dx1, dx2)
        dmu = self.dmu.get()
        self.dx1.set(dmu)
        self.dx2.set(dmu)

    @compilable
    def reset(self):
        zeros = jnp.zeros(self.shape)
        self.x1.set(zeros)
        self.x2.set(zeros)
        self.outputs.set(zeros)
        self.dmu.set(zeros)
        self.dx1.set(zeros)
        self.dx2.set(zeros)
