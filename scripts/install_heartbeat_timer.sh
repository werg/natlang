#!/bin/sh
# Install the heartbeat's systemd user units for this machine (plans/HEARTBEAT_PROGRAM.md, section 8).
#
#   scripts/install_heartbeat_timer.sh            copy the units to ~/.config/systemd/user and reload; the timer stays disabled
#   scripts/install_heartbeat_timer.sh --enable   also enable and start the timer (process control: the machine's owning agent decides)
#   scripts/install_heartbeat_timer.sh --uninstall  stop and disable the timer and remove the units
#
# Units reference the checkout this script lives in. Run it on each machine from /home/werg/natlang.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/.config/systemd/user"
case "${1:-}" in
  --uninstall)
    systemctl --user disable --now natlang-heartbeat.timer 2>/dev/null || true
    rm -f "$DEST/natlang-heartbeat.service" "$DEST/natlang-heartbeat.timer"
    systemctl --user daemon-reload
    echo "heartbeat units removed"; exit 0;;
  ""|--enable) ;;
  *) echo "usage: $0 [--enable|--uninstall]" >&2; exit 2;;
esac
mkdir -p "$DEST" "$REPO/runs/heartbeat"
for unit in natlang-heartbeat.service natlang-heartbeat.timer; do
  sed "s|%h/natlang|$REPO|g" "$REPO/scripts/systemd/$unit" > "$DEST/$unit"
done
systemctl --user daemon-reload
if [ "${1:-}" = "--enable" ]; then
  systemctl --user enable --now natlang-heartbeat.timer
  systemctl --user list-timers natlang-heartbeat.timer --no-pager
else
  echo "installed, not enabled. To enable: systemctl --user enable --now natlang-heartbeat.timer"
  echo "Try one cycle first: systemctl --user start natlang-heartbeat.service ; then read runs/heartbeat/latest.md"
fi
