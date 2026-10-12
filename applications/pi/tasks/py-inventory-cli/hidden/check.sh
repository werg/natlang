set -e
[ "$(python3 inventory.py stock.csv --below 10)" = "$(printf 'widget\nnut\nwasher')" ]
[ "$(python3 inventory.py stock.csv)" = "widget" ]
[ -z "$(python3 inventory.py stock.csv --below 0)" ]
f=$(mktemp)
printf 'item,qty\nzinc,20\nalpha,2\nmid,19\nbig,200\n' > "$f"
[ "$(python3 inventory.py "$f" --below 100)" = "$(printf 'zinc\nalpha\nmid')" ]
[ "$(python3 inventory.py --below 3 "$f")" = "alpha" ]
rm -f "$f"
