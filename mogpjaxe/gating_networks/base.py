#!/usr/bin/env python3
from abc import ABC, abstractmethod

import tensor_annotations.jax as tjax
import jax.numpy as jnp
from gpjax.base import Module
from gpjax.custom_types import InputData, MeanAndCovariance, NumData
from mogpjaxe.custom_types import NumExperts


class GatingNetworkBase(Module, ABC):
    """Abstract base class for the gating network."""

    @abstractmethod
    def predict_fs(self, Xnew: InputData, **kwargs) -> MeanAndCovariance:
        """Calculates the set of gating function posteriors at Xnew

        :param Xnew: inputs with shape [num_data, input_dim]
        TODO correct dimensions
        :returns: mean and cov with shape [..., num_experts, num_data]
        or
        shape [..., num_experts, num_data, num_data]
        """
        raise NotImplementedError

    @abstractmethod
    def predict_mixing_probs(
        self, Xnew: InputData, **kwargs
    ) -> tjax.Array2[NumExperts, NumData]:
        """Calculates the set of experts mixing probabilities at Xnew :math:`\{\Pr(\\alpha=k | x)\}^K_{k=1}`

        :param Xnew: inputs with shape [num_data, input_dim]
        :returns: a batched array with shape [..., num_experts, num_data]
        """
        raise NotImplementedError
