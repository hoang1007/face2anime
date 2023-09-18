from omegaconf import DictConfig
import importlib


def init_module(cfg: DictConfig, reload: bool = False):
    cfg = cfg.copy()
    target_key = '_target_'
    assert target_key in cfg, f'Key {target_key} is required for module initialization!'

    module, cls = cfg.pop(target_key).rsplit('.', 1)
    module = importlib.import_module(module)
    if reload:
        module = importlib.reload(module)
    return getattr(module, cls)(**cfg)
