#!/usr/bin/env python3
"""Keep Colab forwards on its single SSH connection, reusing an existing master."""
import argparse
import subprocess
import time


def maintain(alias, forwards, *, reuse=True, interval=15, run=subprocess.run, sleep=time.sleep):
    # -O never invokes ProxyCommand: a failed check cannot acquire a second Colab session.
    while reuse and run(['ssh', '-O', 'check', alias], capture_output=True, timeout=20).returncode == 0:
        listening = run(['ss', '-tlnH'], capture_output=True, text=True, timeout=20).stdout
        ports = {line.split()[3].rsplit(':', 1)[-1] for line in listening.splitlines() if len(line.split()) >= 4}
        for forward in forwards:
            local, remote = forward.split(':')
            if local not in ports:
                result = run(['ssh', '-O', 'forward', '-L', f'{local}:127.0.0.1:{remote}', alias], timeout=30)
                if result.returncode:
                    return result.returncode
        sleep(interval)
    command = ['ssh', '-N', '-M', '-o', 'ControlPersist=no', '-o', 'ExitOnForwardFailure=yes']
    for forward in forwards:
        local, remote = forward.split(':')
        command += ['-L', f'{local}:127.0.0.1:{remote}']
    # Stay in the service's cgroup. systemd owns/restarts the newly acquired master.
    return run([*command, alias]).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alias', required=True)
    parser.add_argument('--forward', action='append', default=[])
    parser.add_argument('--no-reuse-master', action='store_true', help='diagnostic opt-out: acquire a new master')
    args = parser.parse_args()
    return maintain(args.alias, args.forward, reuse=not args.no_reuse_master)


if __name__ == '__main__':
    raise SystemExit(main())
