import copy, sys, unittest
sys.path.insert(0, '.')
from app.defaults import DEFAULTS
from app.loader import load_config

PRISTINE = copy.deepcopy(DEFAULTS)

class Hidden(unittest.TestCase):
    def test_no_sharing(self):
        first = load_config(environ={})
        first["log"]["handlers"].append("file")
        first["server"]["port"] = 1
        self.assertEqual(load_config(environ={}), PRISTINE)
        self.assertEqual(DEFAULTS, PRISTINE)

    def test_arguments_untouched(self):
        values = {"extra": {"a": 1}, "log": {"handlers": ["syslog"]}}
        config = load_config(values, environ={"APP_EXTRA__B": "2"})
        self.assertEqual(config["extra"], {"a": 1, "b": "2"})
        config["log"]["handlers"].append("x")
        self.assertEqual(values, {"extra": {"a": 1}, "log": {"handlers": ["syslog"]}})

    def test_types(self):
        config = load_config(environ={"APP_DEBUG": "true", "APP_SERVER__WORKERS": "8", "APP_LOG__LEVEL": "warn", "OTHER": "x"})
        self.assertIs(config["debug"], True)
        self.assertEqual(config["server"]["workers"], 8)
        self.assertEqual(config["log"]["level"], "warn")
        self.assertIs(load_config(environ={"APP_DEBUG": "false"})["debug"], False)
        self.assertNotIn("other", config)

unittest.main(argv=['hidden'])
