import unittest
from datetime import date

from dates import add_months, billing_dates, parse


class Dates(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(add_months(date(2023, 3, 15), 2), date(2023, 5, 15))

    def test_year_rollover(self):
        self.assertEqual(add_months(parse("2023-11-05"), 3), date(2024, 2, 5))

    def test_end_of_month(self):
        self.assertEqual(add_months(date(2023, 1, 31), 1), date(2023, 2, 28))


if __name__ == "__main__":
    unittest.main()
