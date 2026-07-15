from __future__ import annotations

import importlib


def load_method_class(name: str):
    module = importlib.import_module(f"method.{name}")
    class_name = "".join(part.upper() if part == "fcamp" else part.capitalize() for part in name.split("_"))
    if hasattr(module, class_name):
        return getattr(module, class_name)
    factory = getattr(module, "build_method", None)
    if factory is not None:
        return factory
    raise AttributeError(
        f"method/{name}.py must define {class_name} or build_method(cfg, env, simulation_app)"
    )
