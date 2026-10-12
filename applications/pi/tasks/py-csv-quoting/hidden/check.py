import csv, io, sys, unittest
sys.path.insert(0, '.')
from report import to_csv

class Hidden(unittest.TestCase):
    def test_fields(self):
        cases = [
            (["say \"hi\""], '"say ""hi"""\n'),
            (["two\nlines", "x"], '"two\nlines",x\n'),
            (["a,b", "c\"d", "plain"], '"a,b","c""d",plain\n'),
            ([" spaced ", None, 3.5], ' spaced ,,3.5\n'),
            (["\"\""], '""""""\n'),
        ]
        for row, expected in cases:
            self.assertEqual(to_csv([row]), expected, row)

    def test_header_quoted_and_roundtrip(self):
        rows = [["x, y", "q\"uote", "multi\nline"], ["é", "", "1"]]
        text = to_csv(rows, header=["h,1", "h2", "h3"])
        parsed = list(csv.reader(io.StringIO(text, newline='')))
        self.assertEqual(parsed, [["h,1", "h2", "h3"]] + rows)

unittest.main(argv=['hidden'])
