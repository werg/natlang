import unittest

from words import top_words


class TopWords(unittest.TestCase):
    def test_english(self):
        self.assertEqual(top_words("The cat and the hat. THE end!", 2), [("the", 3), ("cat", 1)])


if __name__ == "__main__":
    unittest.main()
