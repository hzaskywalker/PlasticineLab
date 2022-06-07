from .default_config import get_cfg_defaults
from yacs.config import CfgNode

def make_cls_config(self, cfg=None, **kwargs):
    _cfg = self.default_config()
    if cfg is not None:
        if isinstance(cfg, str):
            _cfg.merge_from_file(cfg)
        else:
            _cfg.merge_from_other_cfg(cfg)
    if len(kwargs) > 0:
        _cfg.merge_from_list(sum(list(kwargs.items()), ()))
    return _cfg


def purge_cfg(cfg: CfgNode):
    """Purge configuration for clean logs and logical check.
    Notes:
        If a CfgNode has 'TYPE' attribute,
        its CfgNode children the key of which do not contain 'TYPE' will be removed.
    """
    target_key = cfg.get('TYPE', None)
    removed_keys = []
    for k, v in cfg.items():
        if isinstance(v, CfgNode):
            if target_key is not None and (k != target_key):
                removed_keys.append(k)
            else:
                purge_cfg(v)
    for k in removed_keys:
        del cfg[k]

def load(path=None, opts=None):
    cfg = get_cfg_defaults()
    if path is not None:
        cfg.merge_from_file(path)
    if opts is not None:
        cfg.merge_from_list(opts)
    purge_cfg(cfg)
    cfg.freeze()
    return cfg