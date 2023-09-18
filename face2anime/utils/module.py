from omegaconf import DictConfig
import importlib


def init_module(cfg: DictConfig, reload: bool = False):
    target_key = '_target_'
    if target_key in cfg:
        module, cls = cfg.pop(target_key).rsplit('.', 1)
        module = importlib.import_module(module)
        if reload:
            module = importlib.reload(module)
        return getattr(module, cls)(**cfg)
