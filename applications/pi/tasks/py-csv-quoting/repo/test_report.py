import unittest

from report import to_csv


class ToCsv(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(to_csv([["a", 1], ["b", None]], header=["name", "n"]), "name,n\na,1\nb,\n")

    def test_comma_is_quoted(self):
        self.assertEqual(to_csv([["Smith, John", 42]]), '"Smith, John",42\n')


if __name__ == "__main__":
    unittest.main()
