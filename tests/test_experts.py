#!/usr/bin/env python3
import chex
import jax
from absl.testing import parameterized
from gpjax.kernels import SeparateIndependent, SquaredExponential
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from jax import numpy as jnp
from mogpjaxe.experts import SVGPExperts
from mogpjaxe.gps import SVGPSample

key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

# Data variants
input_dims = [1, 5]
# input_dims = [5]
# output_dims = [4]
output_dims = [1, 4]
# output_dims = [1]
# num_datas = [300, (2, 3, 100)]
num_datas = [300]

# SVGP variants
num_inducings = [30]
whitens = [True]
# whitens = [True, False]
q_diags = [True, False]
# q_diags = [False]

# SVGP.predict_f variants
full_covs = [True, False]
# TODO test full_output_covs=True
# full_output_covs = [True, False]
full_output_covs = [False]

# SVGPexperts variants
num_experts = [3]
num_inducing_samples = [None, 2]
# num_inducing_samples = [None]


class TestSVGPExperts(chex.TestCase):
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
        num_experts=num_experts,
    )
    def test_predict_dist(
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
        num_experts,
    ):
        """Check shapes of output"""
        Xnew = jax.random.uniform(key, shape=(num_data, input_dim))
        # Y = jax.random.uniform(key, shape=(num_data, output_dim))

        def init_svgp_sample():
            mean_function = Constant(output_dim=output_dim)
            likelihood = Gaussian()
            if output_dim > 1:
                kernels = [
                    SquaredExponential(
                        lengthscales=jnp.ones(input_dim, dtype=jnp.float64),
                        variance=2.0,
                    )
                    for _ in range(output_dim)
                ]
                kernel = SeparateIndependent(kernels)
            else:
                kernel = SquaredExponential(
                    lengthscales=jnp.ones(input_dim, dtype=jnp.float64), variance=2.0
                )
            inducing_variable = jax.random.uniform(
                key=key, shape=(num_inducing, input_dim)
            )
            return SVGPSample(
                kernel,
                likelihood,
                inducing_variable,
                mean_function,
                num_latent_gps=output_dim,
                q_diag=q_diag,
                whiten=whiten,
            )

        experts_list = [init_svgp_sample() for _ in range(num_experts)]
        experts = SVGPExperts(experts_list)

        params = experts.get_params()

        def predict_fs(params, Xnew):
            return experts.predict_fs(
                params, Xnew, num_inducing_samples, key, full_cov, full_output_cov
            )

        var_predict_fs = self.variant(predict_fs)
        mean, cov = var_predict_fs(params, Xnew)
        print("mean")
        print(mean.shape)
        print(cov.shape)

        if num_inducing_samples is None:
            if not full_output_cov:
                assert mean.ndim == 3
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
        else:
            if not full_output_cov:
                assert mean.ndim == 4
                assert mean.shape[1] == num_inducing_samples
                assert mean.shape[0] == num_experts
                assert mean.shape[2] == num_data
                assert mean.shape[3] == output_dim
                if full_cov:
                    assert cov.ndim == 5
                    assert cov.shape[1] == num_inducing_samples
                    assert cov.shape[0] == num_experts
                    assert cov.shape[2] == output_dim
                    assert cov.shape[3] == cov.shape[4] == num_data
                else:
                    assert cov.ndim == 4
                    assert cov.shape[1] == num_inducing_samples
                    assert cov.shape[0] == num_experts
                    assert cov.shape[2] == num_data
                    assert cov.shape[3] == output_dim
            else:
                raise NotImplementedError("Need to add tests for full_output_cov=True")
