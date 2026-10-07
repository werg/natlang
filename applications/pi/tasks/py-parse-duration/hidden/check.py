import sys, unittest
sys.path.insert(0, '.')
from duration import parse_duration

class Hidden(unittest.TestCase):
    def test_forms(self):
        for text, seconds in [("2h", 7200), ("45s", 45), ("1d2h", 93600), ("2h30m15s", 9015), ("90m", 5400), (" 3m ", 180), ("1d", 86400)]:
            self.assertEqual(parse_duration(text), seconds, text)
    def test_invalid(self):
        for text in ["", "h", "1x", "2h3", "abc", "-1h", "1h-2m"]:
            with self.assertRaises(ValueError, msg=text):
                parse_duration(text)

unittest.main(argv=['hidden'])
