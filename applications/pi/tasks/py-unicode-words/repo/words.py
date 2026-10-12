"""Word frequency helpers."""
import re
from collections import Counter

WORD = re.compile(r"[A-Za-z]+")


def words(text):
    """The words of text, lowercased, in order."""
    return [word.lower() for word in WORD.findall(text)]


def top_words(text, n=3):
    """The n most common words as (word, count) pairs; ties keep first-seen order."""
    return Counter(words(text)).most_common(n)
