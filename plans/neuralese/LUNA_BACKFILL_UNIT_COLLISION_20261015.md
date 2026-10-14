# Luna backfill unit collision (2026-10-15)

Pop's six-worker Luna pool fell to two active workers although ordinary seed
52070 still had work. The replenisher checked its historical folder-backfill
campaign first. That campaign's manifest uses seed `52070`, the same seed as
the current ordinary campaign, and `complete()` inferred systemd unit names
from the seed alone. It therefore saw ordinary `natlang-pop-luna-s52070-w*`
units as if the finished backfill workers were still running, and returned
before preparing the next campaign. All 128 backfill queue jobs had terminal
results, including an already successful retry of one transport failure.

The fix assigns the backfill its own
`natlang-pop-luna-backfill-s52070-w*` unit namespace for future launches and
completion checks. The ordinary and backfill campaigns can now overlap without
identity collision. The observed backfill completion check is true while
ordinary seed 52070 workers are active. Seed 52071 source preparation started
immediately afterward. The corrected code uses the existing global worker cap
and memory ledger; no data or queue state was migrated.
