from functools import partial

from .discriminator import BaseDiscriminator
from .patchgan import NLayerDiscriminator


Discriminators = {
    'base': BaseDiscriminator,
    'nlayer': NLayerDiscriminator
}


def init_discriminator(name: str, *args, **kwargs):
    """Initializes discriminator"""
    avai_discriminators = list(Discriminators.keys())
    if name not in avai_discriminators:
        raise ValueError('Invalid discriminator name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_discriminators))
    return partial(Discriminators[name], *args, **kwargs)

def get_all_discriminators():
    return tuple(Discriminators.values())
