#!/usr/bin/env python3
"""Main process of an RL controller's ledger unit: runs the controller and relaunches it when it asks for a restart.

    controller_supervisor.py -- python -u -m natlang_neuralese.rl.production run --config CONFIG

The controller exits ``RESTART_EXIT`` (75) after SIGUSR1 (``restart_controller.py --in-place``): collector workers and the
managed actor stay running (rl.detached_processes), and this supervisor starts the controller again, so it runs the
checkout as it is now, inside the same unit, cgroup and ledger claim. Any other exit code ends the unit with that code.
SIGTERM/SIGINT/SIGUSR1 are forwarded to the controller; after SIGTERM/SIGINT the supervisor never relaunches. Stdlib only.
"""
import signal
import subprocess
import sys
import time

RESTART_EXIT = 75


def main(argv):
    if argv[:1] == ['--']:
        argv = argv[1:]
    if not argv:
        raise SystemExit('usage: controller_supervisor.py -- COMMAND...')
    state = {'child': None, 'stopping': False}

    def forward(number):
        def handler(_signal, _frame):
            if number in (signal.SIGTERM, signal.SIGINT):
                state['stopping'] = True
            child = state['child']
            if child is not None and child.poll() is None:
                child.send_signal(number)
        return handler
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(number, forward(number))
    while True:
        state['child'] = subprocess.Popen(argv)
        while True:
            try:
                code = state['child'].wait(timeout=.5)
                break
            except subprocess.TimeoutExpired:
                continue
        if code == RESTART_EXIT and not state['stopping']:
            print(f'controller_supervisor: restarting {argv[-1] if len(argv) < 3 else " ".join(argv[:4])} from the checkout',
                  flush=True)
            continue
        return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
