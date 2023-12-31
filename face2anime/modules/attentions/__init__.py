from .self_attention import SAWrapper
from .attention import AttnBlock
from .spatial_transfomer import SpatialTransformer
from .cross_attention import CAWrapper

Attentions = {
    "SelfAttention": SAWrapper,
    "Transformer": SpatialTransformer,
    "Attention": AttnBlock,
    "CrossAttention":CAWrapper,
}


def init_attention(name):
    """Initializes attention"""
    avai_attentions = list(Attentions.keys())
    if name not in avai_attentions:
        raise ValueError('Invalid attention name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_attentions))
    return Attentions[name]


def get_all_attentions():
    return tuple(Attentions.values())