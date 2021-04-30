#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import tensor_annotations.jax as tjax

# import gpflow as gpf
from gpjax import kullback_leiblers
from gpjax.conditionals import conditional
from gpjax.utilities.ops import sample_mvn_chol, sample_mvn_diag, leading_transpose
from gpjax.custom_types import InputData, MeanAndCovariance, NumData, OutputDim

# from gpflow.conditionals.util import sample_mvn
# from gpflow.config import default_float
from gpjax.kernels import Kernel
from gpjax.likelihoods import Likelihood
from gpjax.mean_functions import MeanFunction
from gpjax.models import SVGP
from mogpjaxe.custom_types import NumExperts, NumSamples


class SVGPSample(SVGP):
    """Extension of GPJax's SVGP class with option to sample inducing variables

    Reimplements predict_f with ability to sample inducing variables.
    If num_inducing_samples is None:
      - Then the standard functionality is achieved, i.e. the inducing variables
        are analytically marginalised.
    If num_inducing_samples is not None:
      - Samples are drawn from the inducing variable distribution and the
        standard GP conditional is called (i.e. q_sqrt=None). The user now has
        the ability marginalise them wherever they please.
    """

    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable,
        mean_function: MeanFunction = None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        # num_data=None,
    ):
        super().__init__(
            kernel,
            likelihood,
            inducing_variable,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
            q_diag=q_diag,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=whiten,
            # num_data=num_data,
        )

    def sample_inducing_variables(
        self, params: dict, key, num_samples: int = None
    ) -> tjax.Array3[NumSamples, NumData, OutputDim]:
        """Returns samples from the inducing variable distribution.

        The distribution is given by,

        .. math::
            q \sim \mathcal{N}(q\_mu, q\_sqrt q\_sqrt^T)

        :param params: dict of paramters containing q_mu and q_sqrt
        :param key: jax.random.PRNGKey()
        :param num_samples: the number of samples to draw
        :returns: samples with shape [num_samples, num_data, output_dim]
        """
        if params["q_sqrt"].ndim == 3:
            samples = sample_mvn_chol(
                key,
                params["q_mu"].T,
                params["q_sqrt"],
                num_samples,
                True,
            )
        elif params["q_sqrt"].ndim == 2:
            samples = sample_mvn_diag(
                key,
                params["q_mu"].T,
                jnp.sqrt(params["q_sqrt"].T),
                num_samples,
            )
        else:
            raise NotImplementedError("Bad dimension for q_sqrt")
        return jnp.transpose(samples, [1, 2, 0])

    def predict_f(
        self,
        params: dict,
        Xnew: InputData,
        key: jnp.DeviceArray = None,
        num_inducing_samples: int = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndCovariance:
        """Compute mean and (co)variance of latent function at Xnew.

        If num_inducing_samples is not None then sample inducing variables instead
        of analytically integrating them. This is required in the mixture of
        experts lower bound.

        :param Xnew: inputs with shape [num_test, input_dim]
        :param num_inducing_samples:
            number of samples to draw from inducing variables distribution.
        :param full_cov:
            If True, draw correlated samples over Xnew. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
        :returns: tuple of Tensors (mean, variance),
            If num_inducing_samples=None,
                means.shape == [num_test, output_dim],
                If full_cov=True and full_output_cov=False,
                    var.shape == [output_dim, num_test, num_test]
                If full_cov=False,
                    var.shape == [num_test, output_dim]
            If num_inducing_samples is not None,
                means.shape == [num_inducing_samples, num_test, output_dim],
                If full_cov=True and full_output_cov=False,
                    var.shape == [num_inducing_samples, output_dim, num_test, num_test]
                If full_cov=False and full_output_cov=False,
                    var.shape == [num_inducing_samples, num_test, output_dim]
        """
        if num_inducing_samples is None:
            mean, cov = conditional(
                params["kernel"],
                Xnew,
                params["inducing_variable"],
                self.kernel,
                f=params["q_mu"],
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                q_sqrt=params["q_sqrt"],
                white=self.whiten,
            )
        else:
            q_mu_samples = self.sample_inducing_variables(
                params, key, num_inducing_samples
            )

            def conditional_wrapper(f_sample):
                return conditional(
                    params["kernel"],
                    Xnew,
                    params["inducing_variable"],
                    self.kernel,
                    f=f_sample,
                    full_cov=full_cov,
                    full_output_cov=full_output_cov,
                    q_sqrt=None,
                    white=self.whiten,
                )

            mean, cov = jax.vmap(conditional_wrapper, out_axes=(0, 0))(q_mu_samples)
        return mean + self.mean_function(params["mean_function"], Xnew), cov
