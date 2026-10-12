import sys, unittest
sys.path.insert(0, '.')
from words import top_words, words

class Hidden(unittest.TestCase):
    def test_accents(self):
        self.assertEqual(top_words("Café café CAFÉ naïve", 5), [("café", 3), ("naïve", 1)])

    def test_scripts(self):
        self.assertEqual(top_words("Ελληνικά ελληνικά, русский!", 5), [("ελληνικά", 2), ("русский", 1)])

    def test_separators(self):
        self.assertEqual(words("abc123def_ghi jkl-mno"), ["abc", "def", "ghi", "jkl", "mno"])
        self.assertEqual(words(""), [])
        self.assertEqual(top_words("42 _ !!", 3), [])

unittest.main(argv=['hidden'])
