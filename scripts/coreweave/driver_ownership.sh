# Lineage driver ownership inside one sandbox (sourced by lfm3_rl_lineage.sh; owner request 2026-11-03).
#
# The provider entrypoint (bootstrap.sh) waits for its first driver; when that driver exits the sandbox completes.
# An in-place handover starts a newer driver beside it. $OWNER (one line: "pid start_ticks state") names the current
# owner: a new driver takes ownership only from a dead owner or one in its failure hold, and any driver that is no
# longer the owner waits, at exit, for the live owner (so the entrypoint's child never ends the sandbox while a newer
# driver trains). The failure hold applies only to the current owner and ends early into the same wait when a newer
# driver takes over.
driver_ticks() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }
driver_owner() { [ -f "$OWNER" ] && cat "$OWNER" || echo "- - -"; }
driver_alive() { [ "$1" != - ] && [ -n "$2" ] && [ "$(driver_ticks "$1")" = "$2" ]; }
driver_take() {  # $1 = state; refuse (exit 75, no hold) while another live owner is not in its hold
  local pid ticks state
  read -r pid ticks state <<< "$(driver_owner)"
  if [ "$pid" != "$$" ] && driver_alive "$pid" "$ticks" && [ "$state" != hold ]; then
    echo "{\"driver\":\"refused\",\"owner\":$pid}"; exit 75
  fi
  echo "$$ $(driver_ticks $$) $1" > "$OWNER.tmp" && mv "$OWNER.tmp" "$OWNER"
  # Register the owner file with the sandbox bootstrap: PID 1 stays up while any registered owner lives, so the
  # sandbox never completes under a live driver even when the entry driver itself is gone.
  if [ -n "${NATLANG_CW_CONTROL_DIR:-}" ]; then
    mkdir -p "$NATLANG_CW_CONTROL_DIR"
    grep -qxF "$OWNER" "$NATLANG_CW_CONTROL_DIR/driver-owners" 2>/dev/null || echo "$OWNER" >> "$NATLANG_CW_CONTROL_DIR/driver-owners"
  fi
}
driver_wait_newer() {  # wait while a live newer driver owns the lineage (it may hand over again)
  local pid ticks state
  while true; do
    read -r pid ticks state <<< "$(driver_owner)"
    { [ "$pid" = "$$" ] || ! driver_alive "$pid" "$ticks"; } && return 0
    echo "{\"driver\":\"waiting_for_newer\",\"owner\":$pid}"
    while driver_alive "$pid" "$ticks"; do sleep "${NATLANG_CW_OWNER_POLL:-30}"; done
  done
}
driver_exit() {  # EXIT trap: $1 = exit status
  local rc=$1 pid ticks state
  read -r pid ticks state <<< "$(driver_owner)"
  if [ "$pid" = "$$" ] && [ "$rc" -ne 0 ] && [ "$rc" -ne 143 ] && [ "$rc" -ne 75 ]; then
    echo "$$ $(driver_ticks $$) hold" > "$OWNER.tmp" && mv "$OWNER.tmp" "$OWNER"
    local left=${NATLANG_CW_HOLD_ON_FAILURE:-0}
    while [ "$left" -gt 0 ]; do  # a newer driver taking over ends the hold early
      read -r pid ticks state <<< "$(driver_owner)"; [ "$pid" != "$$" ] && break
      sleep 5; left=$((left - 5))
    done
  fi
  driver_wait_newer
}
