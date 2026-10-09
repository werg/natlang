#!/usr/bin/env python3
"""Coordination messages between the natlang agents on Pop and DGX.

Every message is one JSON file. The sender stores it locally and copies it to each addressed peer machine over SSH
(a plain file write, so delivery does not depend on the peer's code version). Undeliverable copies wait in the
outbox and are retried by every later invocation. Each reader keeps its own cursor, so several sessions on one
machine never consume each other's messages. Requests stay open until someone replies or closes them. Each machine
also publishes one overwritten status page instead of a stream of status messages.

Procedures and conventions: plans/MACHINE_COORDINATION.md ("Messages").
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import textwrap

REPO = Path(__file__).resolve().parents[1]
HOSTS = {'mltick': 'dgx', 'pop-os': 'pop'}          # hostname -> machine
PEERS = {'dgx': 'dgx', 'pop': 'pop-os'}             # machine -> SSH alias used to reach it
KINDS = ('note', 'request', 'decision', 'reply', 'close')
LEGACY_HEADER = re.compile(r'^## (\d{4}-\d\d-\d\d(?:T[0-9:.]+(?:Z|[+-]\d\d:\d\d)?)?) — (\S+)(?: (.+?))?\s*$', re.M)


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def stamp(moment=None):
    return (moment or now()).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


class Store:
    def __init__(self, repo):
        self.root = repo / '.coordination'
        self.mail, self.outbox = self.root / 'mail', self.root / 'outbox'
        self.cursors, self.status = self.root / 'cursors', self.root / 'status'
        for directory in (self.mail, self.outbox, self.cursors, self.status):
            directory.mkdir(parents=True, exist_ok=True)
        self.machine = os.environ.get('COORD_MACHINE') or HOSTS.get(socket.gethostname(), socket.gethostname())
        peers_file = self.root / 'peers.json'
        self.peers = {**PEERS, **(json.loads(peers_file.read_text()) if peers_file.exists() else {})}
        self.peers.pop(self.machine, None)

    def lock(self):
        handle = (self.root / 'coord.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    def messages(self):
        found = []
        for path in self.mail.glob('*.json'):
            try:
                found.append(json.loads(path.read_text()))
            except (OSError, ValueError) as error:
                print(f'coord: skipping unreadable {path.name}: {error}', file=sys.stderr)
        return sorted(found, key=lambda m: (m.get('received_at', ''), m['id']))

    def write(self, message, directory=None):
        directory = directory or self.mail
        path = directory / f"{message['id']}.json"
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(message, indent=1, ensure_ascii=False) + '\n')
        temporary.rename(path)

    def machine_of(self, address):
        return address if address in ('pop', 'dgx', 'all') else address.split('-', 1)[0]

    def deliver(self, message):
        """Copy a message to every peer machine it addresses; queue it in the outbox where that fails."""
        machines = {self.machine_of(address) for address in message['to']}
        targets = list(self.peers) if 'all' in machines else [m for m in machines if m in self.peers]
        failed = []
        for machine in targets:
            if not self.ssh_write(machine, 'mail', message):
                failed.append(machine)
        if failed:
            self.write({**message, '_pending': failed}, self.outbox)
        return failed

    def ssh_write(self, machine, folder, message, name=None):
        if machine not in self.peers:
            print(f'coord: no SSH alias for machine {machine!r} (add it to .coordination/peers.json)', file=sys.stderr)
            return False
        name = name or f"{message['id']}.json"
        remote = f'/home/werg/natlang/.coordination/{folder}'
        payload = {key: value for key, value in message.items() if key != '_pending'}
        payload.pop('received_at', None)
        if folder == 'mail':
            payload['received_at'] = None   # the receiver's first read assigns local arrival order
        command = (f'mkdir -p {remote} && cat > {remote}/.{name}.tmp && mv {remote}/.{name}.tmp {remote}/{name}')
        try:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', self.peers[machine], command],
                                    input=json.dumps(payload, indent=1, ensure_ascii=False) + '\n', text=True,
                                    capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f'coord: delivery to {machine} failed: {error}', file=sys.stderr)
            return False
        if result.returncode:
            print(f'coord: delivery to {machine} failed: {result.stderr.strip()}', file=sys.stderr)
        return result.returncode == 0

    def flush(self):
        for path in sorted(self.outbox.glob('*.json')):
            message = json.loads(path.read_text())
            remaining = [m for m in message.get('_pending', []) if not self.ssh_write(m, 'mail', message)]
            if remaining:
                self.write({**message, '_pending': remaining}, self.outbox)
            else:
                path.unlink()

    def settle_arrivals(self):
        """Messages copied in by a peer arrive without a local arrival time; assign one in arrival order."""
        for path in sorted(self.mail.glob('*.json')):
            try:
                message = json.loads(path.read_text())
            except ValueError:
                continue
            if not message.get('received_at'):
                message['received_at'] = stamp()
                self.write(message)

    def cursor(self, reader):
        path = self.cursors / f'{reader}.json'
        if path.exists():
            return json.loads(path.read_text())
        others = [json.loads(p.read_text()).get('since', '') for p in self.cursors.glob('*.json')]
        return {'since': max(others, default=''), 'read': []}

    def save_cursor(self, reader, cursor):
        path = self.cursors / f'{reader}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(cursor) + '\n')
        temporary.rename(path)

    def migrate_legacy(self):
        """Import the retired single-file inbox once, as read history; keep the original under archive/."""
        legacy = self.root / 'inbox.md'
        if not legacy.exists():
            return 0
        data = legacy.read_bytes()
        seen = self.root / 'seen.json'
        offset = json.loads(seen.read_text()).get('offset', 0) if seen.exists() else 0
        text = data.decode('utf-8')
        heads = list(LEGACY_HEADER.finditer(text))
        imported, baseline = stamp(), ''
        for index, head in enumerate(heads):
            end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
            # Python 3.10's ISO parser does not accept the UTC Z suffix used by
            # the old inbox. Normalize it explicitly on both machines.
            sent = datetime.datetime.fromisoformat(head.group(1).replace('Z', '+00:00'))
            # Notes past the old reader's offset were never acknowledged: they arrive unread.
            unread = len(text[:head.start()].encode('utf-8')) >= offset
            message = {'id': f"{stamp(sent)}-legacy-{index:04d}", 'from': head.group(2), 'to': [self.machine],
                       'kind': 'note', 'subject': head.group(3) or '(from the retired inbox)', 'body': text[head.end():end].strip(),
                       'sent_at': stamp(sent), 'received_at': imported if unread else stamp(sent), 'legacy': not unread}
            self.write(message)
            if not unread:
                baseline = max(baseline, message['received_at'])
        archive = self.root / 'archive'
        archive.mkdir(exist_ok=True)
        legacy.rename(archive / f'legacy-inbox-{stamp()}.md')   # never overwrite an earlier archive
        for retired in ('seen.json', 'inbox.lock'):
            (self.root / retired).unlink(missing_ok=True)
        self.save_cursor(f'{self.machine}-agent', {'since': baseline, 'read': []})
        return len(heads)


def default_identity(machine):
    if os.environ.get('COORD_AS'):
        return os.environ['COORD_AS']
    if os.environ.get('CLAUDE_CODE_SESSION_ID'):
        return f"{machine}-claude-{os.environ['CLAUDE_CODE_SESSION_ID'][:8]}"
    codex = os.environ.get('CODEX_THREAD_ID') or os.environ.get('CODEX_SESSION_ID')
    if codex:
        return f'{machine}-codex-{codex[:8]}'
    if any(key.startswith('CODEX') for key in os.environ):
        return f'{machine}-codex'
    return f'{machine}-agent'


def addressed(message, store, me):
    return any(a in ('all', store.machine, me) for a in message['to'])


def resolutions(messages):
    return {m['reply_to'] for m in messages if m.get('reply_to') and m['kind'] in ('reply', 'close')}


def render(message, full=True):
    flags = ' [URGENT]' if message.get('urgent') else ''
    reply = f" re {message['reply_to']}" if message.get('reply_to') else ''
    head = (f"── {message['id']}  {message['kind']}{flags}{reply}\n"
            f"   {message['from']} → {', '.join(message['to'])}: {message.get('subject') or ''}")
    if not full:
        return head
    return head + '\n\n' + textwrap.indent(message['body'].rstrip(), '   ') + '\n'


def read_body(args):
    body = sys.stdin.read().strip() if not args.message else args.message
    if not body:
        sys.exit('coord: the message body comes from --message or stdin')
    return body


def make_message(store, me, to, kind, subject, body, reply_to=None, urgent=False):
    moment = now()
    return {'id': f'{stamp(moment)}-{store.machine}-{secrets.token_hex(2)}', 'from': me, 'to': to, 'kind': kind,
            'subject': subject, 'body': body, 'reply_to': reply_to, 'urgent': urgent,
            'sent_at': stamp(moment), 'received_at': stamp(moment)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--as', dest='identity', help='reader/sender name, e.g. dgx-claude-1a2b3c4d (default: derived)')
    sub = parser.add_subparsers(dest='command', required=True)
    send = sub.add_parser('send', help='send a message (body from --message or stdin)')
    send.add_argument('--to', required=True, help='comma-separated: pop, dgx, all or an agent name')
    send.add_argument('--kind', choices=('note', 'request', 'decision'), default='note')
    send.add_argument('--subject', required=True)
    send.add_argument('--urgent', action='store_true')
    send.add_argument('-m', '--message')
    reply = sub.add_parser('reply', help='answer a message; answering a request resolves it unless --keep-open')
    reply.add_argument('id')
    reply.add_argument('--keep-open', action='store_true')
    reply.add_argument('-m', '--message')
    close = sub.add_parser('close', help='resolve a request without further discussion')
    close.add_argument('id')
    close.add_argument('-m', '--message', default='closed')
    inbox = sub.add_parser('inbox', help='unread messages for me and open requests to my machine')
    inbox.add_argument('--ack', action='store_true', help='mark the shown messages read')
    sub.add_parser('brief', help='one-line summary for hooks; silent when there is nothing new')
    sub.add_parser('open', help='open requests on both machines')
    show = sub.add_parser('show', help='print messages by id (prefix match) with their thread')
    show.add_argument('ids', nargs='+')
    log = sub.add_parser('log', help='message history')
    log.add_argument('-n', type=int, default=20)
    log.add_argument('--grep')
    log.add_argument('--from', dest='sender')
    log.add_argument('--legacy', action='store_true', help='include the imported legacy inbox')
    status = sub.add_parser('status', help='show both machines\' status pages; --set replaces mine from stdin')
    status.add_argument('--set', action='store_true')
    sub.add_parser('whoami')
    args = parser.parse_args()

    store = Store(args.repo)
    me = args.identity or default_identity(store.machine)
    with store.lock():
        imported = store.migrate_legacy()
        if imported:
            print(f'coord: imported {imported} legacy inbox notes as read history', file=sys.stderr)
        store.settle_arrivals()
        if args.command not in ('brief', 'whoami'):
            store.flush()
        messages = store.messages()
        by_id = {m['id']: m for m in messages}

        def find(prefix):
            matches = [m for m in messages if m['id'].startswith(prefix)]
            if len(matches) != 1:
                sys.exit(f'coord: {len(matches)} messages match {prefix!r}')
            return matches[0]

        if args.command == 'whoami':
            print(f'{me} on {store.machine}; peers: ' + ', '.join(f'{m} via ssh {h}' for m, h in store.peers.items()))
        elif args.command == 'send':
            to = [t.strip() for t in args.to.split(',') if t.strip()]
            message = make_message(store, me, to, args.kind, args.subject, read_body(args), urgent=args.urgent)
            store.write(message)
            failed = store.deliver(message)
            print(message['id'] + (f'  (queued for {", ".join(failed)})' if failed else ''))
        elif args.command in ('reply', 'close'):
            original = find(args.id)
            kind = 'close' if args.command == 'close' else ('note' if args.keep_open else 'reply')
            body = args.message if args.command == 'close' else read_body(args)
            to = sorted({original['from']} | ({store.machine_of(original['from'])} if original['kind'] == 'request' else set()))
            message = make_message(store, me, to, kind, 'Re: ' + (original.get('subject') or ''), body,
                                   reply_to=original['id'])
            store.write(message)
            failed = store.deliver(message)
            print(message['id'] + (f'  (queued for {", ".join(failed)})' if failed else ''))
        elif args.command in ('inbox', 'brief'):
            cursor = store.cursor(me)
            read = set(cursor['read'])
            resolved = resolutions(messages)
            unread = [m for m in messages if addressed(m, store, me) and m['from'] != me and not m.get('legacy')
                      and m['received_at'] > cursor['since'] and m['id'] not in read]
            open_requests = [m for m in messages if m['kind'] == 'request' and m['id'] not in resolved
                             and addressed(m, store, me) and m not in unread]
            if args.command == 'brief':
                if unread or open_requests:
                    urgent = sum(1 for m in unread if m.get('urgent'))
                    print(f'coord: {len(unread)} unread message(s)' + (f' ({urgent} urgent)' if urgent else '') +
                          f', {len(open_requests)} open request(s) for {store.machine}. '
                          'Run `python3 scripts/coord.py inbox --ack`.')
                return
            for message in unread:
                print(render(message))
            if open_requests:
                print(f'Open requests to {store.machine} (reply or close them):')
                for message in open_requests:
                    print(render(message, full=False))
            if not unread and not open_requests:
                print('coord: nothing new')
            if args.ack and unread:
                store.save_cursor(me, {'since': max(m['received_at'] for m in unread), 'read': []})
        elif args.command == 'open':
            resolved = resolutions(messages)
            for message in messages:
                if message['kind'] == 'request' and message['id'] not in resolved:
                    print(render(message, full=False))
        elif args.command == 'show':
            for prefix in args.ids:
                root = find(prefix)
                while root.get('reply_to') in by_id:
                    root = by_id[root['reply_to']]
                thread, frontier = [root], {root['id']}
                for message in messages:
                    if message.get('reply_to') in frontier:
                        thread.append(message)
                        frontier.add(message['id'])
                for message in thread:
                    print(render(message))
        elif args.command == 'log':
            selected = [m for m in messages if (args.legacy or not m.get('legacy'))
                        and (not args.sender or m['from'] == args.sender)
                        and (not args.grep or re.search(args.grep, m['body'] + (m.get('subject') or ''), re.I))]
            for message in selected[-args.n:]:
                print(render(message))
        elif args.command == 'status':
            if args.set:
                page = {'id': 'status', 'machine': store.machine, 'by': me, 'updated_at': stamp(),
                        'body': sys.stdin.read().strip()}
                (store.status / f'{store.machine}.json').write_text(json.dumps(page, indent=1, ensure_ascii=False) + '\n')
                for machine in store.peers:
                    if not store.ssh_write(machine, 'status', page, name=f'{store.machine}.json'):
                        print(f'coord: status not delivered to {machine}; rerun status --set later', file=sys.stderr)
            for path in sorted(store.status.glob('*.json')):
                page = json.loads(path.read_text())
                print(f"── {page['machine']} status, {page['updated_at']} by {page['by']}\n\n"
                      + textwrap.indent(page['body'], '   ') + '\n')


if __name__ == '__main__':
    main()
