"""Calendar helpers on datetime.date (no time zones involved)."""
from datetime import date


def add_months(day, months):
    """The same day of the month, `months` months later (negative goes back)."""
    index = day.month - 1 + months
    year = day.year + index // 12
    month = index % 12 + 1
    return day.replace(year=year, month=month)


def billing_dates(start, count):
    """The start date and the following monthly billing dates, `count` in all."""
    return [add_months(start, i) for i in range(count)]


def parse(text):
    """Parse 'YYYY-MM-DD'."""
    year, month, day_of_month = (int(part) for part in text.split("-"))
    return date(year, month, day_of_month)
