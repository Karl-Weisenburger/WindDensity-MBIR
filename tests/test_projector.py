"""
Sanity checks on the mbirjax projector as configured by this package.

These guard against a broken or incompatible mbirjax/JAX install: if the
back projector is not the adjoint of the forward projector, MBIR will not
converge to the right answer.
"""
import jax.numpy as jnp
import numpy as np

from conftest import SMALL_RECON_SHAPE


def test_back_projector_is_adjoint_of_forward_projector(small_model_and_weights):
    ct_model, _ = small_model_and_weights
    rng = np.random.default_rng(0)
    x = jnp.array(rng.standard_normal(SMALL_RECON_SHAPE), dtype=jnp.float32)
    sino_shape = np.asarray(ct_model.forward_project(x)).shape
    y = jnp.array(rng.standard_normal(sino_shape), dtype=jnp.float32)

    Ax = np.asarray(ct_model.forward_project(x), dtype=np.float64)
    ATy = np.asarray(ct_model.back_project(y), dtype=np.float64)
    lhs = np.vdot(Ax, np.asarray(y, dtype=np.float64))
    rhs = np.vdot(np.asarray(x, dtype=np.float64), ATy)
    assert abs(lhs - rhs) <= 1e-4 * max(abs(lhs), abs(rhs))


def test_forward_projector_is_linear(small_model_and_weights):
    ct_model, _ = small_model_and_weights
    rng = np.random.default_rng(1)
    a = jnp.array(rng.standard_normal(SMALL_RECON_SHAPE), dtype=jnp.float32)
    b = jnp.array(rng.standard_normal(SMALL_RECON_SHAPE), dtype=jnp.float32)
    lhs = np.asarray(ct_model.forward_project(2.0 * a - 3.0 * b))
    rhs = 2.0 * np.asarray(ct_model.forward_project(a)) - 3.0 * np.asarray(ct_model.forward_project(b))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-4, atol=1e-4 * np.abs(rhs).max())

