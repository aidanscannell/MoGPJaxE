#!/usr/bin/env python3
import typing
from tensor_annotations import axes

# Axes types
NumExperts = typing.NewType("NumExperts", axes.Axis)
NumSamples = typing.NewType("NumSamples", axes.Axis)
