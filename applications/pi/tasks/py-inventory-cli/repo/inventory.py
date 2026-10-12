"""List low-stock items from a CSV file with columns item,qty."""
import argparse
import csv


def low_stock(rows, threshold):
    return [row["item"] for row in rows if row["qty"] < threshold]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="CSV file with a header row item,qty")
    parser.add_argument("--below", default=5, help="list items with qty below this (default 5)")
    args = parser.parse_args(argv)
    with open(args.file, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for name in low_stock(rows, args.below):
        print(name)


if __name__ == "__main__":
    main()
