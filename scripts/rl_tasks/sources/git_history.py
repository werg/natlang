"""Repository-history tasks whose oracle is git itself (fallow register K3; plans/neuralese/TASK_SUPPLY_20261118.md section 19).

Input: the bare repositories under bgkit-data/repos (12,083 GitHub repositories with full history). Each selected
repository contributes one window: the newest `window` commits of its first-parent line (a merge's changes are those
against its first parent; oversize files are marked -diff so git reports them as binary). The model reaches it through a read-only `git` service (`log`, `show`, `diff`, `ls`, `cat`, `grep`) over a JSON snapshot of
that window, bound by SHA-256 like the codebase sources' `repo` asset. The service offers no blame and no pickaxe, so
"which commit added this line" takes a path-limited log and reading patches.

Every answer comes from git commands run on the bare repository at build time (`git blame --porcelain`, `git log`,
`git diff-tree`, `git show`, `git diff`); the snapshot is extracted separately, and scripts/rl_tasks/git_history_solve.mjs
re-solves every case through the service alone and must reach the same answer (the reference check).

Families (one split unit per repository; dev and confirmation repositories are held packets):
  line_origin       the commit that added a given line of a file (blame), a line unique in the file and added once
  line_before       a line that one commit rewrote in place: what it read before
  signature_before  the same for a function or method signature line
  commit_files      the paths one commit (named by its subject) added, modified or deleted
  file_created      the commit that created a file
  last_change       the author name or the UTC date of a file's most recent change
  count_commits     how many commits meet a condition (path, author, author and path, date range, subject prefix,
                    commits that added or deleted a file)
  churn             lines added and deleted in a file over the history the service holds (numstat)
  tree_delta        paths that existed at a named commit and are gone at HEAD, or are new since it
  json_value_at     a JSON field's value in a file at a named commit, where it differs from HEAD
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from ..common import make_case, pick, resolve_str

REPOS = resolve_str('data_hdd', 'bgkit-data', 'repos')
# Assets live on NVMe (latency-critical while collecting); NATLANG_RL_GIT_ASSETS overrides the root.
ASSET_ROOT = resolve_str('data_nvme', 'natlang-rl-git-history-20261120', 'assets')
SOURCE = 'git-history'
MAX_TIP_FILES = 800
MAX_BLOB = 200_000
MAX_PATCH = 250_000
MAX_ASSET = 12_000_000
GIT_ENV = {**{k: v for k, v in os.environ.items() if not k.startswith('GIT_')}, 'LC_ALL': 'C', 'GIT_CONFIG_NOSYSTEM': '1',
           'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_PAGER': 'cat', 'GIT_TERMINAL_PROMPT': '0'}

SERVICE = r'''import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
const FILE = __FILE__;
const SHA256 = __SHA__;
type Change = { path: string; status: 'A' | 'M' | 'D'; old: string | null; new: string | null; added: number | null; deleted: number | null };
type Raw = { sha: string; parents: string[]; author: string; email: string; date: string; message: string; changes: Change[] };
type Snapshot = { repo: string; branch: string; commits: Raw[]; tip: Record<string, string>; blobs: Record<string, string | null>; patches: Record<string, string> };
function snapshot(): Snapshot {
  const cache = ((globalThis as any).__natlangRlGitHistory ??= new Map());
  if (!cache.has(SHA256)) {
    const bytes = readFileSync((process.env.NATLANG_RL_GIT_ASSETS ?? __ROOT__) + '/' + FILE);
    if (createHash('sha256').update(bytes).digest('hex') !== SHA256) throw new Error('git history asset changed: ' + FILE);
    cache.set(SHA256, JSON.parse(bytes.toString('utf8')));
  }
  return cache.get(SHA256);
}
function index(rev: string): number {
  const s = snapshot();
  const m = /^(.*?)((?:[~^]\d*)*)$/.exec(String(rev).trim());
  const base = m ? m[1]! : String(rev);
  let i: number;
  if (base === 'HEAD' || base === s.branch || base === '') i = 0;
  else {
    if (!/^[0-9a-fA-F]{4,40}$/.test(base)) throw new Error(`unknown revision ${rev}: use HEAD, HEAD~n, or a commit hash (or a prefix of at least 4 characters) from log()`);
    const hits = s.commits.map((c, k) => c.sha.startsWith(base.toLowerCase()) ? k : -1).filter(k => k >= 0);
    if (hits.length > 1) throw new Error(`ambiguous revision ${rev}: give more characters of the hash`);
    if (!hits.length) {
      const oldest = s.commits[s.commits.length - 1]!;
      if (oldest.parents.some(p => p.startsWith(base.toLowerCase()))) throw new Error(`${rev} is older than the history this service holds; it starts at ${oldest.sha}`);
      throw new Error(`unknown revision ${rev}: log() lists every commit this history holds`);
    }
    i = hits[0]!;
  }
  for (const step of (m ? m[2]! : '').match(/[~^]\d*/g) ?? []) {
    const n = step.length > 1 ? Number(step.slice(1)) : 1;
    if (step[0] === '^' && n > 1) throw new Error(`${rev}: this history follows first parents only`);
    i += step[0] === '^' ? Math.min(n, 1) : n;
  }
  if (i >= s.commits.length) throw new Error(`${rev} is older than the history this service holds; it starts at ${s.commits[s.commits.length - 1]!.sha}`);
  return i;
}
function tree(i: number): Record<string, string> {
  const s = snapshot();
  const t: Record<string, string> = { ...s.tip };
  for (let k = 0; k < i; k++) for (const c of s.commits[k]!.changes) {
    if (c.old === null) delete t[c.path]; else t[c.path] = c.old;
  }
  return t;
}
function brief(c: Raw): Commit {
  return { sha: c.sha, parents: [...c.parents], author: c.author, email: c.email, date: c.date, subject: c.message.split('\n')[0]! };
}
export type Commit = { sha: string; parents: string[]; author: string; email: string; date: string; subject: string };
export type FileChange = { path: string; status: 'A' | 'M' | 'D'; added: number | null; deleted: number | null };
/** Which repository this is, its branch, and how many commits of its history the service holds (the newest ones). */
export function info(): { repo: string; branch: string; commits: number; oldest: string } {
  const s = snapshot();
  return { repo: s.repo, branch: s.branch, commits: s.commits.length, oldest: s.commits[s.commits.length - 1]!.sha };
}
/**
 * Commits from rev (default HEAD) back to the oldest one held, newest first, like `git log --first-parent`. Filters: path (the commit added, modified or deleted that file, or a file under that directory), author
 * (case-insensitive substring of the author name or email), since/until (dates as YYYY-MM-DD or ISO times, both
 * inclusive; a bare date covers its whole UTC day), grep (a JavaScript regular expression tested on the full message,
 * with flags such as "i"), max (at most this many). Dates are the author dates in UTC.
 */
export function log(options: { rev?: string; path?: string; author?: string; since?: string; until?: string; grep?: string; flags?: string; max?: number } = {}): Commit[] {
  const s = snapshot();
  const start = index(options.rev ?? 'HEAD');
  const dir = options.path ? options.path.replace(/^\.\//, '').replace(/\/+$/, '') : undefined;
  const lo = options.since ? (options.since.length === 10 ? options.since + 'T00:00:00Z' : new Date(options.since).toISOString().replace(/\.\d+Z$/, 'Z')) : undefined;
  const hi = options.until ? (options.until.length === 10 ? options.until + 'T23:59:59Z' : new Date(options.until).toISOString().replace(/\.\d+Z$/, 'Z')) : undefined;
  const re = options.grep !== undefined ? new RegExp(options.grep, (options.flags ?? '').replace(/g/g, '')) : undefined;
  const who = options.author?.toLowerCase();
  const out: Commit[] = [];
  for (const c of s.commits.slice(start)) {
    if (dir !== undefined && dir !== '' && !c.changes.some(f => f.path === dir || f.path.startsWith(dir + '/'))) continue;
    if (who !== undefined && !c.author.toLowerCase().includes(who) && !c.email.toLowerCase().includes(who)) continue;
    if (lo !== undefined && c.date < lo) continue;
    if (hi !== undefined && c.date > hi) continue;
    if (re && !re.test(c.message)) continue;
    out.push(brief(c));
    if (options.max !== undefined && out.length >= options.max) break;
  }
  return out;
}
/** One commit: its metadata, full message, and every file it added, modified or deleted (against its first parent) with added and deleted line counts (null for a binary file), like `git show --numstat`. A rename shows as a deletion and an addition. */
export function show(rev: string): Commit & { message: string; files: FileChange[] } {
  const c = snapshot().commits[index(rev)]!;
  return { ...brief(c), message: c.message, files: c.changes.map(f => ({ path: f.path, status: f.status, added: f.added, deleted: f.deleted })) };
}
/** The unified patch a commit made against its first parent (`git diff parent commit`), optionally only the part for one file. */
export function diff(rev: string, path?: string): string {
  const s = snapshot();
  const text = s.patches[s.commits[index(rev)]!.sha] ?? '';
  if (path === undefined) return text;
  const p = path.replace(/^\.\//, '');
  const parts = text.split(/(?=^diff --git )/m);
  return parts.filter(part => part.startsWith(`diff --git a/${p} b/${p}\n`)).join('');
}
/** Paths of every file in the tree at rev (default HEAD), sorted; with dir, only the files under that directory. */
export function ls(rev = 'HEAD', dir?: string): string[] {
  const t = tree(index(rev));
  const d = dir?.replace(/^\.\//, '').replace(/\/+$/, '');
  return Object.keys(t).filter(p => !d || p.startsWith(d + '/')).sort();
}
/** The text of one file at rev, like `git show rev:path`. */
export function cat(rev: string, path: string): string {
  const s = snapshot();
  const p = path.replace(/^\.\//, '');
  const blob = tree(index(rev))[p];
  if (blob === undefined) throw new Error(`no file ${p} at ${rev}; ls(rev) lists the paths`);
  const text = s.blobs[blob];
  if (text === null || text === undefined) throw new Error(`${p} is a binary or very large file`);
  return text;
}
/** Every line of the text files at rev (default HEAD) that matches a JavaScript regular expression (flags such as "i"), with its file and 1-based line number, like `git grep -n`. */
export function grep(pattern: string, rev = 'HEAD', flags = ''): { path: string; line: number; text: string }[] {
  const s = snapshot();
  const re = new RegExp(pattern, flags.replace(/g/g, ''));
  const t = tree(index(rev));
  const out: { path: string; line: number; text: string }[] = [];
  for (const path of Object.keys(t).sort()) {
    const text = s.blobs[t[path]!];
    if (typeof text !== 'string') continue;
    const lines = text.split('\n');
    if (lines.length && lines[lines.length - 1] === '') lines.pop();
    for (let i = 0; i < lines.length; i++) if (re.test(lines[i]!)) out.push({ path, line: i + 1, text: lines[i]! });
  }
  return out;
}
'''

INTRO = ('git is a read-only git service over the newest {w} commits of the {repo} repository\'s {branch} branch, '
         'following first parents (like git log --first-parent): a merge commit lists both parents, and its changes '
         'and patch are those against its first parent. Dates are author dates in UTC. ')
INSTRUCTIONS = {
    'line_origin': ['File {path} contains this line at HEAD:\n\n    {line}\n\nWhich commit added that line (the commit '
                    'whose patch introduced it in this exact form)? Return the commit\'s full 40-character hash.',
                    'At HEAD, {path} has the line `{line}`. Find the commit that introduced this line as it now reads, '
                    'and return that commit\'s full hash.'],
    'line_before': ['In {path}, one commit rewrote a single line in place, and since then the line reads:\n\n    {line}\n\n'
                    'What did that line read just before that commit? Return the earlier line without leading or '
                    'trailing whitespace.',
                    'The line `{line}` of {path} (as it is at HEAD) replaced an earlier version of itself in one '
                    'commit. Return the line as it read before that commit, trimmed of surrounding whitespace.'],
    'signature_before': ['In {path}, a commit changed the signature line of a function or method; at HEAD it reads:\n\n'
                         '    {line}\n\nWhat was the signature line before that commit? Return it without leading or '
                         'trailing whitespace.',
                         'The definition line `{line}` in {path} was edited in place by one commit. What did this '
                         'definition line read before that edit? Return it trimmed of surrounding whitespace.'],
    'commit_files': ['Which files did the commit with the subject "{subject}" change? List every path it added, '
                     'modified or deleted (a rename counts as deleting the old path and adding the new one), relative '
                     'to the repository root.',
                     'List the paths of all files that the commit titled "{subject}" added, modified or deleted. Return '
                     'paths relative to the repository root; a renamed file appears as its old and its new path.'],
    'file_created': ['Which commit created the file {path}? Return the commit\'s full 40-character hash.',
                     'The file {path} was added to the repository by one of the commits this history holds. Return that '
                     'commit\'s full hash.'],
    'last_change_author': ['Who made the most recent change to {path}? Return the author name exactly as the commit '
                           'records it.',
                           'Find the latest commit that changed {path} and return its author\'s name as recorded.'],
    'last_change_date': ['On which date (UTC) was {path} last changed? Return it as YYYY-MM-DD.',
                         'Return the UTC author date (YYYY-MM-DD) of the most recent commit that changed {path}.'],
    'count_commits': ['How many of the commits this history holds {cond}? Return the number.',
                      'Count the commits in this history that {cond}. Return the count as a number.'],
    'churn': ['Over all the commits this history holds, how many lines were added to {path} and how many were '
              'deleted, as each commit\'s line counts (git numstat) record them? Return {{added, deleted}}, the totals.',
              'Sum the added and the deleted line counts of {path} over every commit in this history that changed it. '
              'Return {{added, deleted}}.'],
    'tree_delta_removed': ['Which file paths existed at the commit with the subject "{subject}" but no longer exist at '
                           'HEAD? Return the paths relative to the repository root.',
                           'Compare the tree of the commit titled "{subject}" with HEAD. Return every path that the '
                           'older tree has and HEAD lacks.'],
    'tree_delta_added': ['Which file paths exist at HEAD that did not exist at the commit with the subject "{subject}"? '
                         'Return the paths relative to the repository root.',
                         'Compare the tree of the commit titled "{subject}" with HEAD. Return every path that HEAD has '
                         'and the older tree lacks.'],
    'json_value_at': ['What was the value of {field} in {path} at the commit with the subject "{subject}" (the file as '
                      'that commit left it)? Return the value as a string.',
                      'Read {path} as the commit titled "{subject}" left it and return its {field} value as a string.'],
}
RETURNS = {'line_origin': 'string', 'line_before': 'string', 'signature_before': 'string', 'commit_files': 'string[]',
           'file_created': 'string', 'last_change_author': 'string', 'last_change_date': 'string',
           'count_commits': 'number', 'churn': '{ added: number; deleted: number }', 'tree_delta_removed': 'string[]',
           'tree_delta_added': 'string[]', 'json_value_at': 'string'}
FAMILY = {'last_change_author': 'last_change', 'last_change_date': 'last_change', 'tree_delta_removed': 'tree_delta',
          'tree_delta_added': 'tree_delta'}
LEVEL = {'line_origin': 2, 'line_before': 2, 'signature_before': 2, 'commit_files': 1, 'file_created': 1,
         'last_change_author': 1, 'last_change_date': 1, 'count_commits': 2, 'churn': 2, 'tree_delta_removed': 2,
         'tree_delta_added': 2, 'json_value_at': 2}
PATCH_FAMILIES = {'line_origin', 'line_before', 'signature_before'}
NEURALESE = {
    'helper': 'read_patch', 'args': {'question': 'string', 'commit': 'string', 'patch': 'string'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read patch, the changes commit made to a repository, for what question needs, and write a private '
                   'note: the added or removed lines that bear on it with their file paths, or that the patch has none.',
    'root_note': 'read_patch gives a private note on one commit\'s patch. You may narrow the candidate commits with '
                 'git.log, read their patches with read_patch and combine the notes with your own exact checks.',
}
SIGNATURE = re.compile(r'^\s*(?:export\s+)?(?:default\s+)?(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?(?:static\s+)?'
                       r'(?:def|function|fn|func|sub|proc)\b|^\s*(?:public|private|protected|internal)\s[^=;]*\(')


_ATTRS: dict[str, str] = {}  # repository -> attributes file marking oversize paths -diff (git then treats them as binary)
BIG_LINES = 4000


def git(repo, *args, binary=False):
    """Output of one git command; text output must be UTF-8 (a UnicodeDecodeError skips the repository)."""
    attrs = _ATTRS.get(str(repo))
    out = subprocess.run(['git', '-C', str(repo), '-c', 'core.quotepath=off', '-c', 'log.mailmap=false',
                          *(['-c', f'core.attributesFile={attrs}'] if attrs else []), *args], capture_output=True,
                         env=GIT_ENV, check=True, timeout=300).stdout
    return out if binary else out.decode('utf-8')


def utc(epoch):
    return dt.datetime.fromtimestamp(int(epoch), dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _candidates(root, limit, seed):
    owners = sorted((e.name for e in os.scandir(root) if e.is_dir()),
                    key=lambda o: hashlib.sha256(f'{seed}|{o}'.encode()).hexdigest())
    for owner in owners[:limit]:
        for entry in sorted(os.scandir(Path(root) / owner), key=lambda e: e.name):
            if entry.is_dir() and not entry.name.endswith('.partial') and (Path(entry.path) / 'HEAD').exists():
                yield Path(entry.path), f'{owner}/{entry.name}'
                break


def _window(repo, window, min_window):
    """The newest `window` commits of HEAD's first-parent line, as [sha, parents...] rows."""
    lines = git(repo, 'rev-list', '--first-parent', '--parents', '-n', str(window), 'HEAD').split('\n')
    rows = [line.split() for line in lines if line.strip()]
    return rows if len(rows) >= min_window else None


def _cat_batch(repo, shas, check=False):
    shas = sorted(shas)
    if not shas:
        return {}
    proc = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch-check' if check else '--batch'],
                          input=('\n'.join(shas) + '\n').encode(), capture_output=True, env=GIT_ENV, check=True,
                          timeout=300)
    data, pos, out = proc.stdout, 0, {}
    for sha in shas:
        header_end = data.index(b'\n', pos)
        size = int(data[pos:header_end].decode().split()[2])
        if check:
            out[sha] = size
            pos = header_end + 1
            continue
        body = data[header_end + 1:header_end + 1 + size]
        pos = header_end + 1 + size + 1
        text = None
        if size <= MAX_BLOB and b'\0' not in body:
            try:
                text = body.decode('utf-8')
            except UnicodeDecodeError:
                text = None
        out[sha] = text
    return out


def _diff_args(row):
    return [row[1], row[0]] if len(row) > 1 else ['--root', row[0]]


def extract(repo, name, window=40, min_window=25, scratch=None):
    """The snapshot of one repository's window, or None when it does not fit the source's bounds.

    Paths whose content anywhere in the window is larger than MAX_BLOB, or whose change in one commit exceeds
    BIG_LINES lines, are marked `-diff` in an attributes file that every later git command of this repository reads
    (registered in _ATTRS), so git itself reports them as binary: no patch text, no line counts, no blame questions."""
    _ATTRS.pop(str(repo), None)
    run = _window(repo, window, min_window)
    if not run:
        return None
    tip = run[0][0]
    tip_tree = {}
    for item in git(repo, 'ls-tree', '-r', '-z', '--full-tree', tip).split('\0'):
        if not item:
            continue
        meta, path = item.split('\t', 1)
        mode, kind, sha = meta.split()
        if kind != 'blob' or '\n' in path or '\t' in path:
            return None
        tip_tree[path] = sha
    if not 5 <= len(tip_tree) <= MAX_TIP_FILES:
        return None
    z = '0' * 40
    raws, big = {}, set()
    for row in run:
        changes = {}
        raw = git(repo, 'diff-tree', '-r', '-z', '--no-renames', '--raw', '--no-commit-id', *_diff_args(row)).split('\0')
        for k in range(0, len(raw) - 1, 2):
            head, path = raw[k], raw[k + 1]
            if not head.startswith(':'):
                return None
            mode_a, mode_b, old, new, status = head[1:].split()
            if mode_a.startswith('160') or mode_b.startswith('160') or status not in 'AMD' or '\n' in path or '\t' in path:
                return None
            changes[path] = {'path': path, 'status': status, 'old': None if old == z else old,
                             'new': None if new == z else new, 'added': None, 'deleted': None}
        for line in git(repo, 'diff-tree', '-r', '--no-renames', '--numstat', '--no-commit-id', '-z',
                        *_diff_args(row)).split('\0'):
            if line.strip():
                added, deleted, path = line.split('\t', 2)
                if added != '-' and int(added) + int(deleted) > BIG_LINES:
                    big.add(path)
        raws[row[0]] = changes
    blob_paths = defaultdict(set)
    for path, sha in tip_tree.items():
        blob_paths[sha].add(path)
    for changes in raws.values():
        for c in changes.values():
            for sha in (c['old'], c['new']):
                if sha:
                    blob_paths[sha].add(c['path'])
    sizes = _cat_batch(repo, blob_paths, check=True)
    for sha, size in sizes.items():
        if size > MAX_BLOB:
            big |= blob_paths[sha]
    if big:
        if any(re.search(r'[\s*?\[\]\\"!#]', p) for p in big):
            return None
        if any(paths & big and paths - big for paths in blob_paths.values()):
            return None  # one blob both under a -diff path and a normal path
        attrs = Path(scratch or '/tmp') / f'git-history-attrs-{hashlib.sha256(name.encode()).hexdigest()[:16]}'
        attrs.write_text(''.join(f'/{p} -diff\n' for p in sorted(big)))
        _ATTRS[str(repo)] = str(attrs)
    fmt = '%H%x00%P%x00%an%x00%ae%x00%at%x00%B%x1e'
    commits, patches = [], {}
    for row in run:
        sha = row[0]
        meta = git(repo, 'show', '-s', f'--format={fmt}', sha).rstrip('\n').rstrip('\x1e')
        _, parents, author, email, at, message = meta.split('\0', 5)
        changes = raws[sha]
        for line in git(repo, 'diff-tree', '-r', '--no-renames', '--numstat', '--no-commit-id', '-z',
                        *_diff_args(row)).split('\0'):
            if not line.strip():
                continue
            added, deleted, path = line.split('\t', 2)
            if path in changes:
                changes[path]['added'] = None if added == '-' else int(added)
                changes[path]['deleted'] = None if deleted == '-' else int(deleted)
        patch = (git(repo, 'diff', '--no-renames', '--no-color', '-U3', row[1], sha) if len(row) > 1 else
                 git(repo, 'show', '--no-renames', '--no-color', '--format=', '-U3', sha))
        if len(patch) > MAX_PATCH:
            return None
        patches[sha] = patch
        commits.append({'sha': sha, 'parents': parents.split(), 'author': author, 'email': email, 'date': utc(at),
                        'message': message.strip('\n'), 'changes': sorted(changes.values(), key=lambda c: c['path'])})
    texts = _cat_batch(repo, [sha for sha, paths in blob_paths.items() if not paths & big])
    texts.update({sha: None for sha, paths in blob_paths.items() if paths & big})
    branch = git(repo, 'symbolic-ref', '--short', 'HEAD').strip() or 'main'
    licence = next((p for p in sorted(tip_tree) if re.fullmatch(r'(?i)(licen[cs]e|copying)(\.(md|txt|rst))?', p)), None)
    first = (texts.get(tip_tree[licence]) or '').strip().split('\n')[0].strip()[:100] if licence else ''
    snap = {'repo': name, 'branch': branch, 'commits': commits, 'tip': tip_tree, 'blobs': texts, 'patches': patches,
            'license': f'{licence}: {first}' if licence else 'no license file in the repository tree',
            'binary_attributes': sorted(big)}
    if len(json.dumps(snap, ensure_ascii=False)) > MAX_ASSET:
        return None
    return snap


# ------------------------------------------------------------------ questions; every gold value is git output


def _trees(snap):
    """Tree (path -> blob) after each window commit, newest first: the service's own reconstruction, used only to
    choose questions; every answer is then taken from git."""
    t = dict(snap['tip'])
    out = []
    for c in snap['commits']:
        out.append(dict(t))
        for f in c['changes']:
            if f['old'] is None:
                t.pop(f['path'], None)
            else:
                t[f['path']] = f['old']
    return out


def _unique_subjects(snap):
    subjects = Counter(c['message'].split('\n')[0] for c in snap['commits'])
    out = {}
    for i, c in enumerate(snap['commits']):
        s = c['message'].split('\n')[0]
        if subjects[s] == 1 and 12 <= len(s) <= 100 and len(s.split()) >= 2 and '"' not in s:
            others = [o for o in subjects if o != s]
            if not any(s in o for o in others):
                out[i] = s
    return out


def _added_lines(patch, path):
    """Stripped '+' lines of one file's part of a git patch."""
    out, inside = [], False
    for line in patch.split('\n'):
        if line.startswith('diff --git '):
            inside = line == f'diff --git a/{path} b/{path}'
        elif inside and line.startswith('+') and not line.startswith('+++'):
            out.append(line[1:].strip())
    return out


def _single_line_rewrites(patch, path):
    """{new stripped line: old stripped line} for change runs of exactly one '-' followed by one '+' in one file."""
    out, inside, run = {}, False, []

    def close():
        if len(run) == 2 and run[0][0] == '-' and run[1][0] == '+':
            out[run[1][1:].strip()] = run[0][1:].strip()
    for line in patch.split('\n'):
        if line.startswith('diff --git '):
            close(); run = []
            inside = line == f'diff --git a/{path} b/{path}'
            continue
        if not inside or line.startswith(('+++', '---', '@@', '\\')):
            if line.startswith('@@'):
                close(); run = []
            continue
        if line.startswith(('+', '-')):
            run.append(line)
        else:
            close(); run = []
    close()
    return out


def _good_line(text):
    return 15 <= len(text) <= 140 and len(re.findall(r'[A-Za-z]{2,}', text)) >= 2 and '`' not in text


def _blame(repo, tip, path):
    """[(commit, line text)] for each line of path at tip, from git blame --porcelain."""
    out, current = [], None
    for line in git(repo, 'blame', '--porcelain', tip, '--', path).split('\n'):
        if re.match(r'^[0-9a-f]{40} \d+ \d+', line):
            current = line.split()[0]
        elif line.startswith('\t'):
            out.append((current, line[1:]))
    return out


def _range(snap):
    oldest = snap['commits'][-1]
    return [f'{oldest["parents"][0]}..{snap["commits"][0]["sha"]}'] if oldest['parents'] else [snap['commits'][0]['sha']]


def _log(repo, snap, *args):
    """git log over exactly the window's commits; args after '--' are paths."""
    k = args.index('--') if '--' in args else len(args)
    return [line for line in git(repo, 'log', '--first-parent', '--no-renames', *args[:k], *_range(snap), *args[k:])
            .split('\n') if line.strip()]


def questions(repo, snap, rng):
    commits = snap['commits']
    shas = [c['sha'] for c in commits]
    window = set(shas)
    tip = shas[0]
    trees = _trees(snap)
    touched = defaultdict(list)  # path -> [commit index] (newest first)
    for i, c in enumerate(commits):
        for f in c['changes']:
            touched[f['path']].append(i)
    texty = [p for p, b in snap['tip'].items() if isinstance(snap['blobs'].get(b), str)]
    out = []

    # line_origin / line_before / signature_before: blame at tip, cross-checked against the window's patches
    files = [p for p in texty if len(touched.get(p, [])) >= 2]
    rng.shuffle(files)
    origin, before = [], []
    for path in files[:6]:
        blame = _blame(repo, tip, path)
        counts = Counter(text.strip() for _, text in blame)
        added_in = defaultdict(list)
        for i in touched[path]:
            for text in _added_lines(snap['patches'][shas[i]], path):
                added_in[text].append(i)
        for commit, text in blame:
            s = text.strip()
            if commit not in window or counts[s] != 1 or not _good_line(s) or len(set(added_in.get(s, []))) != 1:
                continue
            i = added_in[s][0]
            if shas[i] != commit:
                continue
            status = next(f['status'] for f in commits[i]['changes'] if f['path'] == path)
            others = [shas[j] for j in touched[path] if j != i]
            if others:
                origin.append(('line_origin', {'path': path, 'line': s}, commit,
                               others[:2] + ([shas[i + 1]] if i + 1 < len(shas) and shas[i + 1] not in others[:2] else []),
                               [commit.upper()], 1 + (status == 'M')))
            rewrites = _single_line_rewrites(snap['patches'][commit], path)
            if status == 'M' and s in rewrites and _good_line(rewrites[s]) and rewrites[s].lower() != s.lower():
                old = rewrites[s]
                kind = 'signature_before' if SIGNATURE.search(s) and SIGNATURE.search(old) else 'line_before'
                wrong = [s] + [v for k, v in rewrites.items() if v != old and k != s][:1]
                before.append((kind, {'path': path, 'line': s}, old, wrong, ['  ' + old + ' '], 2))
    rng.shuffle(origin); rng.shuffle(before)
    out += origin[:2]
    sig = [q for q in before if q[0] == 'signature_before']
    out += sig[:1] + [q for q in before if q[0] == 'line_before'][:2 - min(1, len(sig))]

    subjects = _unique_subjects(snap)
    # commit_files
    picks = [i for i in subjects if 2 <= len(commits[i]['changes']) <= 12]
    rng.shuffle(picks)
    lists = [sorted(f['path'] for f in c['changes']) for c in commits]
    for i in picks[:1]:
        gold = sorted(p for p in git(repo, 'show', '--diff-merges=first-parent', '--no-renames', '--name-only',
                                     '--format=', shas[i]).split('\n') if p)
        near = [p for j in (i - 1, i + 1) if 0 <= j < len(commits) for p in lists[j] if p not in gold]
        wrong = [gold[:-1]] + ([gold + near[:1]] if near else []) + [l for l in lists if l != gold][:1]
        out.append(('commit_files', {'subject': subjects[i]}, gold, wrong, [list(reversed(gold))], 1))

    # file_created
    def statuses(p):
        return [f['status'] for i in touched.get(p, []) for f in commits[i]['changes'] if f['path'] == p]
    created = [p for p in texty if len(touched.get(p, [])) >= 2 and statuses(p).count('A') == 1
               and 'D' not in statuses(p)]
    rng.shuffle(created)
    for path in created[:1]:
        gold = _log(repo, snap, '--diff-filter=A', '--format=%H', '--', path)
        if len(gold) == 1 and gold[0] in window:
            others = [shas[j] for j in touched[path] if shas[j] != gold[0]]
            out.append(('file_created', {'path': path}, gold[0], others[:2], [gold[0].upper()], 1))

    # last_change: author (when the file has two or more authors) or date
    authors = {c['author'] for c in commits}
    multi = [p for p in texty if len({commits[i]['author'] for i in touched.get(p, [])}) >= 2]
    rng.shuffle(multi)
    for path in multi[:1]:
        name = git(repo, 'log', '--first-parent', '-1', '--format=%an', tip, '--', path).strip()
        wrong = sorted({commits[i]['author'] for i in touched[path]} - {name})[:2]
        out.append(('last_change_author', {'path': path}, name, wrong, [name.upper()], 1))
    dated = [p for p in texty if len({commits[i]['date'][:10] for i in touched.get(p, [])}) >= 2 and p not in multi[:1]]
    rng.shuffle(dated)
    for path in dated[:1]:
        at = git(repo, 'log', '--first-parent', '-1', '--format=%at', tip, '--', path).strip()
        gold = utc(at)[:10]
        wrong = sorted({commits[i]['date'][:10] for i in touched[path]} - {gold})[:2]
        out.append(('last_change_date', {'path': path}, gold, wrong, [gold + ' '], 1))

    # count_commits: conditions over the window, counted from git log output
    conds = []
    dirs = Counter(p.split('/')[0] for c in commits for p in {f['path'].split('/')[0] + '/' for f in c['changes']
                                                                if '/' in f['path']})
    for d, n in dirs.items():
        if 3 <= n <= len(commits) - 3:
            gold = len(_log(repo, snap, '--format=%H', '--', d))
            conds.append(('dir', f'changed at least one file under the directory {d}/', gold, {'dir': d}))
    if len(authors) >= 2:
        names = Counter(line for line in _log(repo, snap, '--format=%an'))
        for a, n in names.items():
            if any(a.lower() in b.lower() for b in names if b != a):
                continue  # a substring of another author's name reads two ways
            if 2 <= n <= len(commits) - 2:
                conds.append(('author', f'were authored by {a}', n, {'author': a}))
                for d in list(dirs)[:3]:
                    m = len([1 for line in _log(repo, snap, '--format=%an', '--', d) if line == a])
                    if 1 <= m < n:
                        conds.append(('author-dir', f'were authored by {a} and changed a file under {d}/', m, {'author': a, 'dir': d}))
    days = sorted({c['date'][:10] for c in commits})
    if len(days) >= 6:
        a, b = sorted(rng.sample(days[1:-1], 2))
        if a != b:
            stamps = [utc(x)[:10] for x in _log(repo, snap, '--format=%at')]
            n = sum(1 for d in stamps if a <= d <= b)
            if 2 <= n < len(commits):
                conds.append(('dates', f'have an author date from {a} through {b} (UTC, both days included)', n,
                              {'since': a, 'until': b}))
    first = Counter(m.group(0) for m in (re.match(r'^[A-Za-z]+(?:\([^)]*\))?:', s) for s in
                                         _log(repo, snap, '--format=%s')) if m)
    for prefix, n in first.items():
        if 2 <= n <= len(commits) - 2:
            gold = sum(1 for s in _log(repo, snap, '--format=%s') if s.startswith(prefix))
            conds.append(('prefix', f'have a subject that starts with "{prefix}" (exactly, case-sensitive)', gold,
                          {'prefix': prefix}))
    for flt, text in (('A', 'added at least one new file'), ('D', 'deleted at least one file')):
        n = len(_log(repo, snap, f'--diff-filter={flt}', '--format=%H'))
        if 2 <= n <= len(commits) - 2:
            conds.append((f'filter-{flt}', text, n, {'status': flt}))
    rng.shuffle(conds)
    seen_kinds = set()
    for kind, text, gold, extra in conds:
        if kind in seen_kinds or len(seen_kinds) >= 2:
            continue
        seen_kinds.add(kind)
        wrong = [gold + 1, gold - 1, len(commits)] if gold != len(commits) else [gold + 1, gold - 1]
        out.append(('count_commits', {'cond': text, 'kind': kind, **extra}, gold, [w for w in wrong if w != gold and w >= 0],
                    [str(gold)], 2))

    # churn
    churny = [p for p in texty if len(touched.get(p, [])) >= 3]
    rng.shuffle(churny)
    for path in churny[:1]:
        added = deleted = 0
        rows = _log(repo, snap, '--numstat', '--format=', '--', path)
        if any(r.split('\t')[0] == '-' for r in rows):
            continue
        last = None
        for r in rows:
            a, d, p = r.split('\t', 2)
            if p == path:
                added += int(a); deleted += int(d)
                last = last or (int(a), int(d))
        gold = {'added': added, 'deleted': deleted}
        wrong = [{'added': deleted, 'deleted': added}, {'added': added + 1, 'deleted': deleted},
                 {'added': last[0], 'deleted': last[1]}]
        out.append(('churn', {'path': path}, gold, [w for w in wrong if w != gold], [{'deleted': deleted, 'added': added}], 2))

    # tree_delta: paths removed since / added since a named commit
    picks = [i for i in subjects if i >= 3]
    rng.shuffle(picks)
    done = set()
    for i in picks:
        for kind, flt in (('tree_delta_removed', 'D'), ('tree_delta_added', 'A')):
            if kind in done:
                continue
            gold = sorted(p for p in git(repo, 'diff', '--no-renames', '--name-only', f'--diff-filter={flt}',
                                         shas[i], tip).split('\n') if p)
            if 1 <= len(gold) <= 10:
                other = sorted((set(trees[i]) ^ set(trees[0])) - set(gold))
                wrong = [gold[:-1]] if len(gold) > 1 else []
                if other:
                    wrong.append(sorted(gold + other[:1]))
                near_i = sorted(p for p in git(repo, 'diff', '--no-renames', '--name-only', f'--diff-filter={flt}',
                                               shas[i - 1], tip).split('\n') if p)
                if near_i != gold:
                    wrong.append(near_i)
                if not wrong:
                    continue
                out.append((kind, {'subject': subjects[i]}, gold, wrong, [list(reversed(gold))], 2 + (len(gold) > 3)))
                done.add(kind)
        if len(done) == 2:
            break

    # json_value_at: a scalar field of a JSON file at a named commit that differs from HEAD
    jsons = [p for p in texty if p.endswith('.json') and len(touched.get(p, [])) >= 2]
    rng.shuffle(jsons)
    found = False
    for path in jsons:
        if found:
            break
        try:
            now = json.loads(git(repo, 'show', f'{tip}:{path}'))
        except (ValueError, subprocess.CalledProcessError):
            continue
        for i in sorted(touched[path]):
            if i not in subjects or not isinstance(now, dict):
                continue
            try:
                then = json.loads(git(repo, 'show', f'{shas[i]}:{path}'))
            except (ValueError, subprocess.CalledProcessError):
                continue
            if not isinstance(then, dict):
                continue
            fields = []
            for key, value in then.items():
                if isinstance(value, str) and 1 <= len(value) <= 60 and now.get(key) != value and key in now:
                    fields.append(([key], value, now[key]))
                if isinstance(value, dict):
                    for sub, v in value.items():
                        cur = now.get(key, {}).get(sub) if isinstance(now.get(key), dict) else None
                        if isinstance(v, str) and 1 <= len(v) <= 60 and cur is not None and cur != v:
                            fields.append(([key, sub], v, cur))
            if fields:
                keys, gold, head = fields[0]
                field = ' -> '.join(f'"{k}"' for k in keys)
                field = f'the field {field}' if len(keys) == 1 else f'the nested field {field}'
                wrong = [head if isinstance(head, str) else json.dumps(head)]
                if wrong[0].strip().lower() != gold.strip().lower() and gold.lower() not in subjects[i].lower():
                    out.append(('json_value_at', {'path': path, 'subject': subjects[i], 'field': field, 'keys': keys}, gold, wrong,
                                [f' {gold} '], 2))
                    found = True
                break
    return out


def _asset(name, snap, root):
    rel = 'git-history/' + re.sub(r'[^A-Za-z0-9._-]+', '_', name) + '.json'
    target = Path(root) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snap, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if not target.exists() or target.read_text() != text:
        target.write_text(text)
    return rel, hashlib.sha256(target.read_bytes()).hexdigest()


def _service(rel, sha, root):
    return SERVICE.replace('__FILE__', json.dumps(rel)).replace('__SHA__', json.dumps(sha)) \
        .replace('__ROOT__', json.dumps(root))


def _split(name):
    bucket = int(hashlib.sha256(f'git-history-20261120|{name}'.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 86 else 'dev' if bucket < 93 else 'confirmation'


def _spread(qs, per_repo, rng):
    """At most per_repo questions, one per kind before any kind repeats (families vary within a repository)."""
    by_kind = defaultdict(list)
    for q in qs:
        by_kind[q[0]].append(q)
    kinds = sorted(by_kind)
    rng.shuffle(kinds)
    out = []
    while len(out) < per_repo and any(by_kind.values()):
        for kind in kinds:
            if by_kind[kind] and len(out) < per_repo:
                out.append(by_kind[kind].pop(0))
    return out


def cases(repos=REPOS, scan=1500, want=160, window=40, per_repo=7, seed=20261120, out_assets=ASSET_ROOT, report=None):
    out, log = [], []
    accepted = 0
    attrs_dir = Path(out_assets).parent / 'git-attributes'
    attrs_dir.mkdir(parents=True, exist_ok=True)
    for path, name in _candidates(repos, scan, seed):
        if accepted >= want:
            break
        try:
            snap = extract(path, name, window=window, scratch=attrs_dir)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, UnicodeError) as error:
            log.append({'repo': name, 'skip': f'{type(error).__name__}'})
            continue
        if snap is None:
            log.append({'repo': name, 'skip': 'bounds'})
            continue
        rng = random.Random(f'{seed}:{name}')
        try:
            qs = questions(path, snap, rng)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, UnicodeError) as error:
            log.append({'repo': name, 'skip': f'questions: {type(error).__name__}'})
            continue
        if len(qs) < 4:
            log.append({'repo': name, 'skip': f'{len(qs)} questions'})
            continue
        qs = _spread(qs, per_repo, rng)
        accepted += 1
        rel, sha = _asset(name, snap, out_assets)
        service = _service(rel, sha, out_assets)
        split = _split(name)
        unit = f'git:{name}'
        url = f'https://github.com/{name}'
        log.append({'repo': name, 'questions': len(qs), 'split': split, 'commits': len(snap['commits'])})
        intro = INTRO.format(w=len(snap['commits']), repo=name, branch=snap['branch'])
        for kind, params, gold, wrong, aliases, level in qs:
            item = f'{name}:{kind}:{json.dumps(params, sort_keys=True, ensure_ascii=False)}'
            text = pick(INSTRUCTIONS[kind], item).format(**params)
            returns = RETURNS[kind]
            oracle = ({'level': 'normalized', 'normalization': 'answer-list'} if returns == 'string[]' else 'normalized')
            framing = pick(['plain', 'neuralese-helper'], item + '|framing') if kind in PATCH_FAMILIES else 'plain'
            case = make_case(
                source=SOURCE, family=f'git_{FAMILY.get(kind, kind)}', shape=f'git-{kind.replace("_", "-")}',
                upstream=f'github:{name}', upstream_id=item, upstream_split=split, unit=unit, groups=[unit],
                root_name='investigate_history', args={}, returns=returns, instruction=intro + text, inputs={},
                expected=gold, oracle=oracle, level=level, license_=f'{url} ({snap["license"]})',
                services={'git': service}, domain='relational', slice_='inline_placement', wrong=wrong,
                aliases=aliases, revision=snap['commits'][0]['sha'], neuralese=NEURALESE if kind in PATCH_FAMILIES else None,
                framing=framing, split=split,
                neuralese_room='per-commit patch notes from read_patch fanned in to the answer' if kind in PATCH_FAMILIES else '')
            case['curriculum']['external']['git'] = {'repo': name, 'url': url, 'tip': snap['commits'][0]['sha'],
                                                     'window': len(snap['commits']), 'asset': rel, 'asset_sha256': sha,
                                                     'kind': kind, 'params': params}
            out.append(case)
    if report:
        Path(report).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in log))
    return out


def stage(dest, repos=REPOS, scan=1500, min_mb=0.2, max_mb=25.0, seed=20261120):
    """Copy the first repository of each of `scan` hash-ordered owners whose object store is min_mb..max_mb to dest
    (NVMe), as whole sequential file copies: git's random reads on the shared HDD are too slow to build from."""
    import shutil
    dest = Path(dest)
    copied = 0
    for path, name in _candidates(repos, scan, seed):
        size = sum(f.stat().st_size for f in (path / 'objects').rglob('*') if f.is_file()) / 1e6
        target = dest / name
        if not min_mb <= size <= max_mb or target.exists():
            copied += target.exists()
            continue
        stage_dir = target.with_name(target.name + '.partial')
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.copytree(path, stage_dir, symlinks=True)
        stage_dir.rename(target)
        copied += 1
    return copied


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'stage':
        print(stage(sys.argv[2], scan=int(sys.argv[3]) if len(sys.argv) > 3 else 1500))
