"""Parse human durations like '1h30m' into seconds."""
import re

UNITS = {"h": 3600, "m": 60}


def parse_duration(text):
    match = re.fullmatch(r"(\d+)h(\d+)m", text.strip())
    if not match:
        return None
    hours, minutes = match.groups()
    return int(hours) * UNITS["h"] + int(minutes) * UNITS["m"]
