import sys, unittest
from datetime import date
sys.path.insert(0, '.')
from dates import add_months, billing_dates

class Hidden(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(add_months(date(2024, 1, 31), 1), date(2024, 2, 29))
        self.assertEqual(add_months(date(2024, 3, 31), -1), date(2024, 2, 29))
        self.assertEqual(add_months(date(2023, 5, 31), 1), date(2023, 6, 30))
        self.assertEqual(add_months(date(2024, 2, 29), 12), date(2025, 2, 28))
        self.assertEqual(add_months(date(2023, 1, 15), -13), date(2021, 12, 15))
        self.assertEqual(add_months(date(2023, 12, 31), 0), date(2023, 12, 31))

    def test_billing(self):
        self.assertEqual(billing_dates(date(2024, 1, 31), 5),
                         [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31), date(2024, 4, 30), date(2024, 5, 31)])
        self.assertEqual(billing_dates(date(2024, 1, 1), 0), [])

unittest.main(argv=['hidden'])
