#!/usr/bin/env python3
from typing import List

import jax
import jax.numpy as jnp
import tensor_annotations.jax as tjax
from gpjax.custom_types import MeanAndCovariance
from gpjax.kernels import Kernel
from gpjax.likelihoods import Likelihood
from gpjax.mean_functions import MeanFunction
from mogpjaxe.custom_types import NumExperts, NumSamples
from mogpjaxe.experts import ExpertsBase
from mogpjaxe.gps import SVGPSample

InputData = None


# class SVGPExpert(SVGPSample, ExpertBase):
#     """Sparse Variational Gaussian Process Expert.

#     This class inherits the prior_kl() method from the SVGPModel class
#     and implements the predict_dist() method using SVGPModel's predict_y
#     method.
#     """

#     def __init__(
#         self,
#         kernel: Kernel,
#         likelihood: Likelihood,
#         inducing_variable,
#         mean_function: MeanFunction = None,
#         num_latent_gps: int = 1,
#         q_diag: bool = False,
#         q_mu=None,
#         q_sqrt=None,
#         whiten: bool = True,
#     ):
#         super().__init__(
#             kernel,
#             likelihood,
#             inducing_variable,
#             mean_function,
#             num_latent_gps,
#             q_diag,
#             q_mu,
#             q_sqrt,
#             whiten,
#         )

#     def predict_dist(
#         self,
#         params: dict,
#         Xnew: InputData,
#         # num_inducing_samples: int = None,
#         full_cov: bool = False,
#         full_output_cov: bool = False,
#     ) -> MeanAndCovariance:
#         """Returns the mean and (co)variance of the experts prediction at Xnew.

#         :param Xnew: inputs with shape [num_data, input_dim]
#         :param num_inducing_samples:
#             the number of samples to draw from the inducing points joint distribution.
#         :param full_cov:
#             If True, draw correlated samples over the inputs. Computes the Cholesky over the
#             dense covariance matrix of size [num_data, num_data].
#             If False, draw samples that are uncorrelated over the inputs.
#         :param full_output_cov:
#             If True, draw correlated samples over the outputs.
#             If False, draw samples that are uncorrelated over the outputs.
#         :returns: tuple of Tensors (mean, variance),
#             means shape is [num_inducing_samples, num_data, output_dim],
#             if full_cov=False variance tensor has shape
#             [num_inducing_samples, num_data, ouput_dim]
#             and if full_cov=True,
#             [num_inducing_samples, output_dim, num_data, num_data]
#         """
#         mu, cov = self.predict_y(
#             params,
#             Xnew,
#             full_cov=full_cov,
#             full_output_cov=full_output_cov,
#         )
#         return mu, cov


class SVGPExperts(ExpertsBase):
    """Extension of ExpertsBase for a set of SVGP experts.

    Provides an interface between a set of SVGPExpert instances and the
    MixtureOfSVGPExperts class.
    """

    def __init__(self, experts_list: List[SVGPSample] = None):
        """
        :param experts_list: a list of SVGPSample instances.
        """
        super().__init__(experts_list)
        for expert in experts_list:
            # assert isinstance(expert, SVGPExpert)
            assert isinstance(expert, SVGPSample)

    def prior_kls(self, params) -> tjax.Array1[NumExperts]:
        """Returns the set of experts KL divergences as a batched tensor.

        :returns: a Tensor with shape [num_experts,]
        """
        kls = jax.tree_multimap(
            lambda expert, params_: expert.kl(params_), self.experts_list, params
        )
        kls = jnp.stack(kls, axis=-1)
        return kls

    def predict_fs(
        self,
        params: dict,
        Xnew: InputData,
        num_inducing_samples: int = None,
        key: jnp.DeviceArray = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndCovariance:
        """Returns the set experts latent function mean and (co)vars at Xnew.

        :param Xnew: inputs with shape [num_data, input_dim]
        :returns: a tuple of (mean, (co)var) each with shape [..., num_experts, num_data, output_dim]
        """
        expert_fs = jax.tree_multimap(
            lambda expert, params_: expert.predict_f(
                params_,
                Xnew,
                num_inducing_samples,
                key,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
            ),
            self.experts_list,
            params,
        )
        means, covs = [], []
        for dist in expert_fs:
            mean, cov = dist
            means.append(mean)
            covs.append(cov)

        means = jnp.stack(means, axis=0)
        covs = jnp.stack(covs, axis=0)
        return means, covs

    def predict_dists(
        self,
        params: dict,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ):
        """Returns the set of experts predicted dists at Xnew.

        :param Xnew: inputs with shape [num_data, input_dim]
        :returns: a batched tfd.Distribution with batch_shape [..., num_data, output_dim, num_experts]
        """
        expert_dists = jax.tree_multimap(
            lambda expert, params_: expert.predict_y(
                params_, Xnew, full_cov, full_output_cov
            ),
            self.experts_list,
            params,
        )
        means, covs = [], []
        for dist in expert_dists:
            mean, cov = dist
            means.append(mean)
            covs.append(cov)
        means = jnp.stack(means, axis=0)
        covs = jnp.stack(covs, axis=0)
        return mus, covs

    # def likelihoods(self) -> List[Likelihood]:
    #     likelihoods = []
    #     for expert in self.experts_list:
    #         likelihoods.append(expert.likelihood)
    #     return likelihoods

    # def noise_variances(self) -> List[tf.Tensor]:
    #     variances = []
    #     for expert in self.experts_list:
    #         variances.append(expert.likelihood.variance)
    #     return variances
