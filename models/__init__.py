# models/__init__.py
import importlib, logging
log = logging.getLogger(__name__)

_lazy = {
    "OneMinOHLCPolicy": ("models.onemin_policy", "OneMinOHLCPolicy"),
    "OneMinRecurrentHybridPolicy": ("models.onemin_policy", "OneMinRecurrentHybridPolicy"),
    "LongTermOHLCPolicy": ("models.long_policy", "LongTermOHLCPolicy"),
    "MediumTermOHLCPolicy": ("models.medium_policy", "MediumTermOHLCPolicy"),
    "TickPolicy": ("models.tick_policy", "TickPolicy"),  # optional
}

__all__ = list(_lazy.keys())

def __getattr__(name):
    if name not in _lazy:
        raise AttributeError(name)
    mod_name, attr = _lazy[name]
    try:
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr)
        globals()[name] = val  # cache
        return val
    except ModuleNotFoundError:
        log.debug(f"[models] optional module '{mod_name}' not present")
        raise AttributeError(name)
