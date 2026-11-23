# run on the VM through `colab exec` (the kernel channel): end stale sshd sessions (the dead master's), never the listener
import os, signal
for pid in os.listdir('/proc'):
    if pid.isdigit():
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode().strip()
        except OSError:
            continue
        if cmd.startswith('sshd: root@'):
            os.kill(int(pid), signal.SIGTERM)
            print('ended', pid, cmd)
