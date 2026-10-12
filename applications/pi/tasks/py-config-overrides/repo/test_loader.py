import unittest

from app.loader import load_config


class Loader(unittest.TestCase):
    def test_file_values(self):
        config = load_config({"server": {"host": "0.0.0.0"}}, environ={})
        self.assertEqual(config["server"], {"host": "0.0.0.0", "port": 8000, "workers": 2})

    def test_calls_are_independent(self):
        load_config({"log": {"level": "debug"}}, environ={})
        self.assertEqual(load_config(environ={})["log"]["level"], "info")

    def test_env_port_is_int(self):
        config = load_config(environ={"APP_SERVER__PORT": "9000"})
        self.assertEqual(config["server"]["port"], 9000)


if __name__ == "__main__":
    unittest.main()
