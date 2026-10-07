set -e
[ "$(printf 'a b\nc\n' | node bin/count.js --lines)" = "2" ]
[ "$(node bin/count.js --lines sample.txt)" = "4 sample.txt" ]
[ "$(node bin/count.js sample.txt --lines sample.txt | tail -1)" = "8 total" ]
[ "$(node bin/count.js sample.txt)" = "6 sample.txt" ]
[ "$(printf 'a b\nc' | node bin/count.js --lines)" = "2" ]
grep -q -- '--lines' README.md
