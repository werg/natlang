# Source from unit launchers (systemd-run, timers, ledger waiters): the user manager's PATH lacks ~/.local/bin, so a
# unit found /usr/bin/node v18, where the ts-host runtime fails (no crypto.randomUUID; util.parseArgs allowNegative).
# Units admitted through scripts/memory_ledger.py get the same PATH automatically.
#   . /home/werg/natlang/scripts/natlang_env.sh     (or: sh -c '. scripts/natlang_env.sh && exec CMD')
export PATH="$HOME/.local/bin:$PATH"
