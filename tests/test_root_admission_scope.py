import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "root_admission_scope.py"
SPEC = importlib.util.spec_from_file_location("root_admission_scope", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RootAdmissionScopeTests(unittest.TestCase):
    def test_accepts_false_or_exact_integer_zero(self):
        self.assertTrue(MODULE.no_new_world_credit(False))
        self.assertTrue(MODULE.no_new_world_credit(0))

    def test_rejects_true_nonzero_and_non_integer_zero(self):
        for value in (True, 1, -1, 0.0, "0", None):
            with self.subTest(value=value):
                self.assertFalse(MODULE.no_new_world_credit(value))


if __name__ == "__main__":
    unittest.main()
