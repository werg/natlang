"""Recursive dictionary merge."""


def merge(base, override):
    """Merge override into base: nested dicts merge key by key, other values replace."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base
