#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from gpjax.base import Module
from mogpjaxe.experts import SVGPExperts
from mogpjaxe.gating_networks import SVGPGatingNetworkBase
from mogpjaxe.models import MixtureOfExperts


class MixtureOfSVGPExperts(MixtureOfExperts, Module):
    """Mixture of SVGP experts using stochastic variational inference.

    Implemention of a mixture of Gaussian process (GPs) experts method where
    the gating network is also implemented using GPs.
    The model is trained with stochastic variational inference by exploiting
    the factorization achieved by sparse GPs.

    :param gating_network: an instance of the GatingNetworkBase class with
                            the predict_mixing_probs(Xnew) method implemented.
    :param experts: an instance of the SVGPExperts class with the
                    predict_dists(Xnew) method implemented.
    :param num_inducing_samples: the number of samples to draw from the
                                 inducing point distributions during training.
    :param num_data: the number of data points.
    """

    def __init__(
        self,
        gating_network: SVGPGatingNetworkBase,
        experts: SVGPExperts,
        # num_data: int,
        # num_inducing_samples: int = 0,
    ):
        assert isinstance(gating_network, SVGPGatingNetworkBase)
        assert isinstance(experts, SVGPExperts)
        super().__init__(gating_network, experts)
        # self.num_inducing_samples = num_inducing_samples
        # self.num_data = num_data

    def get_params(self):
        experts_params = self.experts.get_params()
        gating_network_params = self.gating_network.get_params()
        return {"experts": experts_parms, "gating_network": gating_network_params}
