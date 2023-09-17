from .face import FaceDataset
from .scenery import SceneryDataset

__datasets = {
    'face': FaceDataset,
    'scenery': SceneryDataset,
}


def init_dataset(name, **kwargs):
    """Initializes a dataset."""
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_datasets))
    return __datasets[name](**kwargs)