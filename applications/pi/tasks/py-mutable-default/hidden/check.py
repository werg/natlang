import sys, unittest
sys.path.insert(0, '.')
from cart import Cart

class Hidden(unittest.TestCase):
    def test_independent(self):
        carts = [Cart() for _ in range(3)]
        for i, cart in enumerate(carts):
            cart.add("x", i + 1)
        self.assertEqual([c.total_quantity() for c in carts], [1, 2, 3])
        self.assertEqual(Cart().items, [])

    def test_initial_items(self):
        cart = Cart([("a", 2)])
        cart.add("b")
        self.assertEqual(cart.total_quantity(), 3)
        self.assertEqual(Cart().total_quantity(), 0)

    def test_rejects_zero(self):
        with self.assertRaises(ValueError):
            Cart().add("a", 0)

unittest.main(argv=['hidden'])
