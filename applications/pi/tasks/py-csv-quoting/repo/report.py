"""Write rows of values as CSV text (one line per row, '\n' line endings)."""


def format_field(value):
    """Render one value; None becomes an empty field."""
    if value is None:
        return ""
    return str(value)


def to_csv(rows, header=None):
    lines = []
    if header:
        lines.append(",".join(format_field(name) for name in header))
    for row in rows:
        lines.append(",".join(format_field(value) for value in row))
    return "\n".join(lines) + "\n"
