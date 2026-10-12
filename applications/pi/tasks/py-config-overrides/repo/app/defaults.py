"""Built-in configuration; file values and APP_* environment variables override it."""

DEFAULTS = {
    "debug": False,
    "server": {"host": "127.0.0.1", "port": 8000, "workers": 2},
    "log": {"level": "info", "handlers": ["console"]},
}
