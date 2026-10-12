import unittest

from cart import Cart


class CartTest(unittest.TestCase):
    def test_add(self):
        cart = Cart()
        cart.add("apple", 2)
        cart.add("pear")
        self.assertEqual(cart.total_quantity(), 3)
        self.assertEqual(cart.names(), ["apple", "pear"])

    def test_new_carts_are_empty(self):
        first = Cart()
        first.add("apple")
        second = Cart()
        self.assertEqual(second.total_quantity(), 0)


if __name__ == "__main__":
    unittest.main()
