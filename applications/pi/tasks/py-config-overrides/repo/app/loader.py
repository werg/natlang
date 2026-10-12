"""Load the configuration: DEFAULTS, then file values, then the environment."""
import os

from .defaults import DEFAULTS
from .merge import merge

PREFIX = "APP_"


def parse_env(environ):
    """APP_SERVER__PORT=9000 becomes {"server": {"port": "9000"}}."""
    result = {}
    for name, raw in environ.items():
        if not name.startswith(PREFIX):
            continue
        path = name[len(PREFIX):].lower().split("__")
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = raw
    return result


def load_config(file_values=None, environ=None):
    config = merge(DEFAULTS, file_values or {})
    env = parse_env(os.environ if environ is None else environ)
    return merge(config, env)
