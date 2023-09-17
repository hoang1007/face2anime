from functools import partial

from .generator import BaseGenerator
from .resnet import ResnetGenerator


Generators = {
    'base': BaseGenerator,
    'resnet': ResnetGenerator
}


def init_generator(name: str, *args, **kwargs):
    """Initializes generator"""
    avai_generators = list(Generators.keys())
    if name not in avai_generators:
        raise ValueError('Invalid generator name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_generators))
    return partial(Generators[name], *args, **kwargs)

def get_all_generators():
    return tuple(Generators.values())
