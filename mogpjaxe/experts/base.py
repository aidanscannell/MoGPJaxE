#!/usr/bin/env python3
import abc
from typing import List

import jax
from gpjax.base import Module

InputData = None


# class ExpertBase(Module, abc.ABC):
#     """Abstract base class for an individual expert.

#     Each subclass that inherits ExpertBase should implement the predict_dist()
#     method that returns the individual experts prediction at an input.
#     """

#     @abc.abstractmethod
#     def predict_dist(self, params: dict, Xnew: InputData, **kwargs):
#         # def predict_dist(self, Xnew: InputData, **kwargs) -> tfd.Distribution:
#         """Returns the individual experts prediction at Xnew.

#         TODO: this does not return a tfd.Distribution

#         :param Xnew: inputs with shape [num_test, input_dim]
#         :returns: an instance of a TensorFlow Distribution
#         """
#         raise NotImplementedError


class ExpertsBase(Module, abc.ABC):
    """Abstract base class for a set of experts.

    Provides an interface between ExpertBase and MixtureOfExperts.
    Each subclass that inherits ExpertsBase should implement the predict_dists()
    method that returns the set of experts predictions at an input (as a
    batched TensorFlow distribution).
    """

    def __init__(self, experts_list: List = None):
        """
        :param experts_list: A list of experts
        """
        super().__init__()
        assert isinstance(
            experts_list, list
        ), "experts_list should be a list of experts objects"
        # for expert in experts_list:
        #     assert isinstance(expert, ExpertBase)
        self.num_experts = len(experts_list)
        self.experts_list = experts_list

    def get_params(self) -> dict:
        return jax.tree_map(lambda expert: expert.get_params(), self.experts_list)

    @abc.abstractmethod
    def predict_dists(self, params: dict, Xnew: InputData, **kwargs):
        """Returns the set of experts predicted dists at Xnew.

        :param Xnew: inputs with shape [num_test, input_dim]
        :returns: a batched tfd.Distribution with batch_shape [..., num_test, output_dim, num_experts]
        """
        raise NotImplementedError
