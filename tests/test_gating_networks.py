#!/usr/bin/env python3
import chex
import jax
from absl.testing import parameterized
from gpjax.kernels import SeparateIndependent, SquaredExponential
from gpjax.likelihoods import Bernoulli
from gpjax.mean_functions import Zero
from jax import numpy as jnp
from mogpjaxe.gating_networks.svgp import SVGPGatingNetwork

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

# Data variants
input_dims = [1, 5]
input_dims = [5]
# num_datas = [300, (2, 3, 100)]
num_datas = [300]

# SVGP variants
num_inducings = [30]
whitens = [True]
# whitens = [True, False]
q_diags = [True, False]
q_diags = [False]

# SVGP.predict_f variants
full_covs = [True, False]
full_covs = [True]
# TODO test full_output_covs=True
# full_output_covs = [True, False]
full_output_covs = [False]

# SVGPexperts variants
num_gating_functions = [1, 2, 3]
num_gating_functions = [1]
num_inducing_samples = [None, 2]
num_inducing_samples = [None]
# likelihoods = [Bernoulli(), Softmax()]


class TestSVGPGatingNetwork(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        input_dim=input_dims,
        num_gating_functions=num_gating_functions,
        num_data=num_datas,
        num_inducing=num_inducings,
        q_diag=q_diags,
        whiten=whitens,
        full_cov=full_covs,
        full_output_cov=full_output_covs,
        num_inducing_samples=num_inducing_samples,
    )
    def test_predict_mixing_probs(
        self,
        input_dim,
        num_gating_functions,
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
        # Y = jax.random.uniform(key, shape=(num_data, output_dim))

        def init_gating_function():
            mean_function = Zero(output_dim=num_gating_functions)
            if num_gating_functions > 1:
                # likelihood = Softmax()
                kernels = [
                    SquaredExponential(
                        lengthscales=jnp.ones(input_dim, dtype=jnp.float64),
                        variance=2.0,
                    )
                    for _ in range(num_gating_functions)
                ]
                kernel = SeparateIndependent(kernels)
            else:
                likelihood = Bernoulli()
                kernel = SquaredExponential(
                    lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
                )
            inducing_variable = jax.random.uniform(
                key=key, shape=(num_inducing, input_dim)

            return SVGPGatingNetwork(
                kernel,
                likelihood,
                inducing_variable,
                mean_function,
                num_latent_gps=num_gating_functions,
                q_diag=q_diag,
                whiten=whiten,
            )

        gating_function_list = [
            init_gating_function() for _ in range(num_gating_functions)
        ]
        gating_network = SVGPGatingNetwork(gating_function_list)

        params = gating_network.get_params()

        def predict_mixing_probs(params, Xnew):
            return gating_network.predict_mixing_probs(
                params, Xnew, num_inducing_samples, key, full_cov, full_output_cov
            )

        var_predict_mixing_probs = self.variant(predict_mixing_probs)
        mixing_probs = var_predict_mixing_probs(params, Xnew)
        print("mixing_probs")
        print(mixing_probs.shape)

        if not full_output_cov:
            assert mixing_probs.ndim == 3
            assert mean.shape[0] == num_experts
            assert mean.shape[1] == num_data
            assert mean.shape[2] == output_dim
            if full_cov:
                assert cov.ndim == 4
                assert cov.shape[0] == num_experts
                assert cov.shape[1] == output_dim
                assert cov.shape[2] == cov.shape[3] == num_data
            else:
                assert cov.ndim == 3
                assert cov.shape[0] == num_experts
                assert cov.shape[1] == num_data
                assert cov.shape[2] == output_dim
        else:
            raise NotImplementedError("Need to add tests for full_output_cov=True")
