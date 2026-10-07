set -e
[ -f data/customers.csv ] && [ -f data/footer.txt ]
[ ! -e build/old.html ]
grep -q 'Welcome home.' build/index.html && grep -q '(c) 2026 Example' build/about.html
