#!/usr/bin/env python3
import chex
import jax
from absl.testing import parameterized
from jax import numpy as jnp
from mogpjaxe.gps.svgp import SVGPSample
from gpjax.kernels import SeparateIndependent, SquaredExponential
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant


key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

# Data variants
input_dims = [1, 5]
# input_dims = [5]
# output_dims = [4]
output_dims = [1, 4]
# output_dims = [4]
# num_datas = [300, (2, 3, 100)]
num_datas = [300]

# SVGP variants
num_inducings = [30]
whitens = [True]
# whitens = [True, False]
q_diags = [True, False]
# q_diags = [False]
# q_diags = [True]

# SVGP.predict_f variants
full_covs = [True, False]
# TODO test full_output_covs=True
# full_output_covs = [True, False]
full_output_covs = [False]
# num_inducing_samples = [None, 1, 5]
num_inducing_samples = [None, 5]
# num_inducing_samples = [5]


class TestSVGPSample(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        input_dim=input_dims,
        output_dim=output_dims,
        num_data=num_datas,
        num_inducing=num_inducings,
        q_diag=q_diags,
        whiten=whitens,
        full_cov=full_covs,
        full_output_cov=full_output_covs,
        num_inducing_samples=num_inducing_samples,
    )
    def test_predict_f(
        self,
        input_dim,
        output_dim,
        num_data,
        num_inducing,
        q_diag,
        whiten,
        full_cov,
        full_output_cov,
        num_inducing_samples,
    ):
        """Check shapes of output"""
        Xnew = jax.random.uniform(key, shape=(num_data, input_dim))

        mean_function = Constant(output_dim=output_dim)
        likelihood = Gaussian()
        if output_dim > 1:
            kernels = [
                SquaredExponential(
                    lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
                )
                for _ in range(output_dim)
            ]
            kernel = SeparateIndependent(kernels)
        else:
            kernel = SquaredExponential(
                lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
            )
        inducing_variable = jax.random.uniform(key=key, shape=(num_inducing, input_dim))

        svgp = SVGPSample(
            kernel,
            likelihood,
            inducing_variable,
            mean_function,
            num_latent_gps=output_dim,
            q_diag=q_diag,
            whiten=whiten,
        )

        params = svgp.get_params()

        def predict_f(params, Xnew):
            return svgp.predict_f(
                params, Xnew, key, num_inducing_samples, full_cov, full_output_cov
            )

        var_predict_f = self.variant(predict_f)
        mean, cov = var_predict_f(params, Xnew)
        print("mean")
        print(mean.shape)
        print(cov.shape)

        if num_inducing_samples is None:
            if not full_output_cov:
                assert mean.ndim == 2
                assert mean.shape[0] == num_data
                assert mean.shape[1] == output_dim
                if full_cov:
                    assert cov.ndim == 3
                    assert cov.shape[0] == output_dim
                    assert cov.shape[1] == cov.shape[2] == num_data
                else:
                    assert cov.ndim == 2
                    assert cov.shape[0] == num_data
                    assert cov.shape[1] == output_dim
            else:
                raise NotImplementedError("Need to add tests for full_output_cov=True")
        else:
            if not full_output_cov:
                assert mean.ndim == 3
                assert mean.shape[0] == num_inducing_samples
                assert mean.shape[1] == num_data
                assert mean.shape[2] == output_dim
                if full_cov:
                    assert cov.ndim == 4
                    assert cov.shape[0] == num_inducing_samples
                    assert cov.shape[1] == output_dim
                    assert cov.shape[2] == cov.shape[3] == num_data
                else:
                    assert cov.ndim == 3
                    assert cov.shape[0] == num_inducing_samples
                    assert cov.shape[1] == num_data
                    assert cov.shape[2] == output_dim
            else:
                raise NotImplementedError("Need to add tests for full_output_cov=True")
