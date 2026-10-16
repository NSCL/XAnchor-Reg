from xanchor.anchors import Anchors, compute_anchors, merge_anchors
from xanchor.registration import LEVELS, RegistrationResult, register

__all__ = ['Anchors', 'compute_anchors', 'merge_anchors', 'register', 'RegistrationResult', 'LEVELS', 'DescriptorEncoder']


def __getattr__(name):
    if name == 'DescriptorEncoder':
        from xanchor.encoder import DescriptorEncoder
        return DescriptorEncoder
    raise AttributeError(name)
