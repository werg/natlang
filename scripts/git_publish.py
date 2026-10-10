#!/usr/bin/env python3
"""Publish a set of paths to origin/main from the shared checkout without stashing or touching other files.

The shared checkout holds many sessions' uncommitted (and staged) work at once. `git stash`, `git merge --autostash`
and `git pull --rebase --autostash` sweep all of it into a stash; a conflicting re-apply leaves the stash behind,
conflict markers in other sessions' files and their staged entries unstaged. This tool never does that:

  1. `git fetch origin`;
  2. builds the commit in a temporary index seeded from origin/main (`git read-tree origin/main`) holding the
     caller's paths' worktree content. What to publish is decided against origin/main: a path whose worktree
     content differs from origin/main's is published, including content committed locally but never pushed. The
     caller's base is the merge base of the local HEAD and origin/main (the local HEAD itself when it is behind);
     a path upstream changed since that base is merged 3-way (`git merge-file` of base / origin/main / worktree);
     any conflict aborts before anything is pushed;
  3. `git commit-tree` with parent origin/main, `git push origin <sha>:main`; a non-fast-forward rejection refetches
     and rebuilds (up to --retries times);
  4. advances the shared checkout without stashing: paths whose worktree already holds the new content get that
     index entry, then `git read-tree -m -u HEAD <new>` (a two-tree fast-forward that updates only clean paths and
     keeps staged entries) and `git update-ref HEAD <new> <old>`. If another session's dirty or staged file is
     changed upstream, the checkout is not advanced at all and the blocking paths are reported (exit 3); the push
     already happened, and `--sync-only` advances later.

Usage:
  scripts/git_publish.py -m "Subject" [-m "Body paragraph"] [--trailer "Key: value"] PATH...
  scripts/git_publish.py --dry-run -m "Subject" PATH...      # show the plan; push and change nothing
  scripts/git_publish.py --sync-only                         # fetch and fast-forward the checkout only

A deleted path is published as a deletion. A directory expands to its tracked and untracked (not ignored) files.
`scripts/check-main.sh --quick` runs first unless --no-check. Exit codes: 0 published (or nothing to do),
1 error / conflict (nothing pushed), 2 usage, 3 published but the checkout was not advanced.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NOT_ADVANCED = 0, 1, 2, 3
NULL_SHA = "0" * 40


class PublishError(Exception):
    pass


class Git:
    def __init__(self, root: Path, index_file: str | None = None):
        self.root = root
        self.index_file = index_file

    def run(self, *args, input: bytes | None = None, check: bool = True, env_extra: dict | None = None):
        env = dict(os.environ)
        env.pop("GIT_INDEX_FILE", None)
        if self.index_file:
            env["GIT_INDEX_FILE"] = self.index_file
        if env_extra:
            env.update(env_extra)
        proc = subprocess.run(["git", *args], cwd=self.root, input=input, capture_output=True, env=env)
        if check and proc.returncode != 0:
            raise PublishError(f"git {' '.join(args)} failed ({proc.returncode}): "
                               f"{proc.stderr.decode(errors='replace').strip()}")
        return proc

    def out(self, *args, **kw) -> str:
        return self.run(*args, **kw).stdout.decode().strip()

    def rev(self, name: str) -> str | None:
        proc = self.run("rev-parse", "--verify", "--quiet", name + "^{commit}", check=False)
        return proc.stdout.decode().strip() or None

    def entry(self, commit: str, path: str) -> tuple[str, str] | None:
        """(mode, blob) of path in commit, or None."""
        line = self.out("ls-tree", "-z", commit, "--", path).rstrip("\0")
        if not line:
            return None
        meta, _ = line.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind != "blob":
            raise PublishError(f"{path} is a {kind} in {commit[:10]}, not a file")
        return mode, sha

    def blob(self, sha: str) -> bytes:
        return self.run("cat-file", "blob", sha).stdout

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.run("merge-base", "--is-ancestor", a, b, check=False).returncode == 0


def worktree_entry(git: Git, path: str) -> tuple[str, str] | None:
    """(mode, blob) of the worktree file as git would store it (clean filters applied), writing the blob."""
    full = git.root / path
    try:
        st = os.lstat(full)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode):
        target = os.readlink(full).encode()
        return "120000", git.out("hash-object", "-w", "--no-filters", "--stdin", input=target)
    if stat.S_ISDIR(st.st_mode):
        raise PublishError(f"{path} is a directory")
    mode = "100755" if st.st_mode & stat.S_IXUSR else "100644"
    return mode, git.out("hash-object", "-w", f"--path={path}", "--", str(full))


def expand_paths(git: Git, args: list[str], cwd: Path) -> list[str]:
    paths: set[str] = set()
    for arg in args:
        full = (cwd / arg).resolve() if not os.path.isabs(arg) else Path(arg).resolve()
        # resolve() follows a symlink; keep the link itself when the argument names one.
        candidate = (cwd / arg) if not os.path.isabs(arg) else Path(arg)
        if candidate.is_symlink():
            full = candidate.parent.resolve() / candidate.name
        try:
            rel = full.relative_to(git.root).as_posix()
        except ValueError:
            raise PublishError(f"{arg} is outside the repository {git.root}")
        if rel in ("", "."):
            raise PublishError("refusing to publish the whole repository; name the paths")
        if full.is_dir() and not full.is_symlink():
            listed = git.out("ls-files", "-z", "-co", "--exclude-standard", "--", rel).split("\0")
            listed += git.out("ls-tree", "-r", "-z", "--name-only", "HEAD", "--", rel).split("\0")
            found = {p for p in listed if p}
            if not found:
                raise PublishError(f"{arg}: no files")
            paths |= found
        else:
            paths.add(rel)
    for p in sorted(paths):
        if p.split("/")[-1] == "node_modules" or "/node_modules/" in f"/{p}/":
            raise PublishError(f"{p}: never publish node_modules")
    return sorted(paths)


def build_message(messages: list[str], trailers: list[str], git: Git) -> bytes:
    text = "\n\n".join(m.strip("\n") for m in messages if m.strip()) + "\n"
    if trailers:
        args = ["interpret-trailers"]
        for t in trailers:
            args += ["--trailer", t]
        text = git.run(*args, input=text.encode()).stdout.decode()
    return text.encode()


def merge3(git: Git, path: str, base: tuple[str, str], upstream: tuple[str, str], mine: tuple[str, str]):
    """3-way merge of blobs; returns the merged blob sha or raises on conflict."""
    with tempfile.TemporaryDirectory(prefix="git-publish-merge-") as tmp:
        files = {}
        for name, (_, sha) in (("mine", mine), ("base", base), ("upstream", upstream)):
            files[name] = Path(tmp) / name
            files[name].write_bytes(git.blob(sha))
        proc = git.run("merge-file", "-p", "-L", f"{path} (worktree)", "-L", f"{path} (base)", "-L",
                       f"{path} (origin/main)", str(files["mine"]), str(files["base"]), str(files["upstream"]),
                       check=False)
        if proc.returncode != 0:
            n = proc.returncode if proc.returncode > 0 else "?"
            raise PublishError(f"{path}: upstream changed it since your base and the 3-way merge has {n} "
                               "conflict(s); nothing was pushed. Merge origin/main's version into your file by hand "
                               f"(git show origin/main:{path}) and publish again.")
        return git.out("hash-object", "-w", "--no-filters", "--stdin", input=proc.stdout)


def plan_paths(git: Git, base: str, upstream: str, paths: list[str]):
    """For each path: (action, new entry or None, note). Raises on conflict. `base` is the merge base of the local
    HEAD and `upstream`: content the caller committed locally but never pushed differs from it and is published."""
    plan = []
    for path in paths:
        b = git.entry(base, path)
        u = git.entry(upstream, path)
        w = worktree_entry(git, path)
        if w is None and b is None and u is None:
            raise PublishError(f"{path}: not in the worktree, HEAD or origin/main")
        if w == u:
            new, note = u, "already upstream"
        elif u == b:
            new, note = w, "worktree"
        elif w is None or u is None or b is None:
            if w == b:  # caller did not touch it: keep upstream
                new, note = u, "upstream (unchanged locally)"
            else:
                what = "deleted" if u is None else ("added" if b is None else "changed")
                raise PublishError(f"{path}: upstream {what} it since your base and your worktree differs; "
                                   "nothing was pushed. Reconcile it by hand and publish again.")
        elif w == b:
            new, note = u, "upstream (unchanged locally)"
        else:
            sha = merge3(git, path, b, u, w)
            mode = w[0] if w[0] != b[0] else u[0]
            new, note = (mode, sha), "merged with upstream"
        plan.append((path, new, note))
    return plan


def build_commit(root: Path, upstream: str, plan, message: bytes) -> tuple[str | None, str]:
    """Commit (sha or None if no change) whose tree is upstream's plus the plan; returns (commit, tree)."""
    with tempfile.TemporaryDirectory(prefix="git-publish-index-") as tmp:
        idx = Git(root, str(Path(tmp) / "index"))
        idx.run("read-tree", upstream)
        for path, new, _ in plan:
            if new is None:
                idx.run("update-index", "--force-remove", "--", path)
            else:
                idx.run("update-index", "--add", "--cacheinfo", f"{new[0]},{new[1]},{path}")
        tree = idx.out("write-tree")
    git = Git(root)
    if tree == git.out("rev-parse", upstream + "^{tree}"):
        return None, tree
    commit = git.out("commit-tree", tree, "-p", upstream, input=message)
    return commit, tree


def index_entries(git: Git) -> dict[str, tuple[str, str, int]]:
    out = {}
    for rec in git.out("ls-files", "-s", "-z").split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode, sha, stage = meta.split()
        out[path] = (mode, sha, int(stage))
    return out


def advance_checkout(git: Git, new: str, *, dry_run: bool = False, write_back: list[str] = ()) -> tuple[bool, str]:
    """Fast-forward HEAD, index and clean worktree paths to `new` (or to a merge of `new` into local commits);
    never overwrite dirty or staged foreign work. `write_back`: the caller's merged paths, whose worktree file is
    replaced by the target's content (it contains the caller's change) before the dirty check.

    Returns (advanced, message)."""
    old = git.rev("HEAD")
    if old is None:
        raise PublishError("the checkout has no HEAD")
    if old == new or git.is_ancestor(new, old):
        return True, f"checkout already at or past {new[:10]}"
    gitdir = Path(git.out("rev-parse", "--absolute-git-dir"))
    if any((gitdir / n).exists() for n in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD")):
        return False, "a merge, rebase or cherry-pick is in progress in the checkout; not advanced"
    if not git.is_ancestor(old, new):
        # HEAD holds local commits not on the remote (another session's, unpushed). Merge in the remote like
        # `git merge` would, but in object space: the local commits stay local, nothing is stashed.
        proc = git.run("merge-tree", "--write-tree", "--no-messages", old, new, check=False)
        if proc.returncode != 0:
            return False, (f"the checkout's HEAD {old[:10]} has local commits that conflict with {new[:10]}; "
                           "not advanced. Their owner must merge them.")
        tree = proc.stdout.decode().split()[0]
        if dry_run:
            return True, f"would merge {new[:10]} into the checkout's local commits at {old[:10]}"
        new = git.out("commit-tree", tree, "-p", old, "-p", new,
                      input=f"Merge {new[:10]} (origin/main) into the shared checkout's local commits\n\n"
                            "Made by scripts/git_publish.py (no stash).\n".encode())
    for path in write_back:
        if git.entry(new, path) is not None:
            (git.root / path).write_bytes(git.run("cat-file", "--filters", f"{new}:{path}").stdout)
    changed = [p for p in git.out("diff-tree", "-r", "-z", "--name-only", "--no-renames", old, new).split("\0") if p]
    index = index_entries(git)
    blocking, adopt, unindex = [], [], []
    for path in changed:
        h, m = git.entry(old, path), git.entry(new, path)
        i = index.get(path)
        if i is not None and i[2] != 0:
            blocking.append(f"{path} (unmerged)")
            continue
        i2 = (i[0], i[1]) if i else None
        w = worktree_entry(git, path)
        if w == m:
            # The worktree already holds the new content (the caller's published paths, an earlier publish).
            if i2 == m:
                continue
            if i2 == h or i2 is None:
                if m is None:
                    unindex.append(path)
                else:
                    adopt.append((path, m))
                continue
            blocking.append(f"{path} (staged content differs from both HEAD and {new[:10]})")
            continue
        if i2 == m:
            blocking.append(f"{path} (staged as the new content but the worktree differs)")
            continue
        if i2 != h:
            blocking.append(f"{path} (staged changes)")
            continue
        if w != h:
            blocking.append(f"{path} (uncommitted changes)")
    if blocking:
        return False, ("not advancing the checkout: other work touches paths that changed upstream:\n  "
                       + "\n  ".join(blocking)
                       + "\nCommit or publish them (or let their owner do it), then run scripts/git_publish.py --sync-only.")
    if dry_run:
        return True, f"would advance the checkout {old[:10]} -> {new[:10]} ({len(changed)} paths)"
    for path, (mode, sha) in adopt:
        git.run("update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}")
    for path in unindex:
        git.run("update-index", "--force-remove", "--", path)
    for attempt in range(5):
        proc = git.run("read-tree", "-m", "-u", old, new, check=False)
        if proc.returncode == 0:
            break
        err = proc.stderr.decode(errors="replace")
        if "index.lock" in err and attempt < 4:
            time.sleep(0.5 * (attempt + 1))
            continue
        return False, f"git read-tree -m -u refused (checkout not advanced):\n{err.strip()}"
    proc = git.run("update-ref", "-m", "git_publish: advance", "HEAD", new, old, check=False)
    if proc.returncode != 0:
        return False, ("index and worktree advanced but HEAD moved concurrently; HEAD not updated: "
                       + proc.stderr.decode(errors="replace").strip())
    return True, f"checkout advanced {old[:10]} -> {new[:10]}"


def fetch(git: Git, remote: str, branch: str) -> str:
    git.run("fetch", "--quiet", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}")
    sha = git.rev(f"refs/remotes/{remote}/{branch}")
    if sha is None:
        raise PublishError(f"{remote}/{branch} does not exist")
    return sha


def before_push(attempt: int) -> None:
    """Test hook (monkeypatched by tests to simulate a concurrent push)."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("paths", nargs="*", help="files or directories to publish (worktree content)")
    ap.add_argument("-m", "--message", action="append", default=[], help="message paragraph (repeatable)")
    ap.add_argument("-F", "--file", help="read the message from a file")
    ap.add_argument("--trailer", action="append", default=[], help='"Key: value" trailer (repeatable)')
    ap.add_argument("--dry-run", action="store_true", help="show the plan; push and change nothing")
    ap.add_argument("--no-check", action="store_true", help="skip scripts/check-main.sh --quick")
    ap.add_argument("--sync-only", action="store_true", help="only fetch and fast-forward the checkout")
    ap.add_argument("--no-advance", action="store_true", help="push but leave the checkout as it is")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--retries", type=int, default=5, help="rebuilds after a non-fast-forward rejection")
    ap.add_argument("-C", "--repo", default=".", help="run as if started in this directory")
    a = ap.parse_args(argv)

    cwd = Path(a.repo).resolve()
    try:
        root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True,
                                   check=True).stdout.decode().strip())
    except subprocess.CalledProcessError:
        print(f"git_publish: {cwd} is not in a git repository", file=sys.stderr)
        return EXIT_USAGE
    git = Git(root)
    try:
        if a.sync_only:
            upstream = fetch(git, a.remote, a.branch)
            ok, msg = advance_checkout(git, upstream, dry_run=a.dry_run)
            print(msg)
            return EXIT_OK if ok else EXIT_NOT_ADVANCED
        if not a.paths:
            ap.error("name the paths to publish")
        messages = list(a.message)
        if a.file:
            messages.append(Path(a.file).read_text())
        if not any(m.strip() for m in messages):
            ap.error("a commit message is required (-m or -F)")
        paths = expand_paths(git, a.paths, Path.cwd() if a.repo == "." else cwd)
        message = build_message(messages, a.trailer, git)

        if not a.no_check and not a.dry_run:
            check = root / "scripts" / "check-main.sh"
            if check.exists():
                print("git_publish: scripts/check-main.sh --quick", flush=True)
                if subprocess.run([str(check), "--quick"], cwd=root).returncode != 0:
                    print("git_publish: check-main.sh --quick failed; nothing pushed (fix it, or --no-check)",
                          file=sys.stderr)
                    return EXIT_ERROR

        head = git.rev("HEAD")
        if head is None:
            raise PublishError("the checkout has no HEAD")
        for attempt in range(a.retries + 1):
            upstream = fetch(git, a.remote, a.branch)
            # The caller's base: where the local HEAD and upstream diverged (HEAD itself when it is behind), so that
            # named paths' content committed locally but never pushed counts as the caller's change.
            base = git.out("merge-base", head, upstream)
            if attempt == 0 and base != head:
                ahead = git.out("rev-list", "--count", f"{upstream}..{head}")
                print(f"git_publish: note: the checkout's HEAD has {ahead} commit(s) not on {a.remote}/{a.branch}; "
                      "only the named paths' content is published (their local commits included), not the commits",
                      flush=True)
            plan = plan_paths(git, base, upstream, paths)
            commit, _ = build_commit(root, upstream, plan, message)
            for path, new, note in plan:
                print(f"  {'delete' if new is None else 'publish'} {path} ({note})")
            if commit is None:
                print(f"git_publish: nothing to publish; {a.remote}/{a.branch} already has these contents")
                if a.dry_run:  # a dry run never changes HEAD, index or worktree
                    print(advance_checkout(git, upstream, dry_run=True)[1])
                    return EXIT_OK
                new_head = upstream
                break
            if a.dry_run:
                print(git.out("diff", "--stat", upstream, commit))
                print(f"git_publish: dry run; would push {commit[:10]} onto {upstream[:10]}")
                ok, msg = advance_checkout(git, commit, dry_run=True)
                print(msg)
                return EXIT_OK
            before_push(attempt)
            proc = git.run("push", "--porcelain", a.remote, f"{commit}:refs/heads/{a.branch}", check=False)
            if proc.returncode == 0:
                git.run("update-ref", f"refs/remotes/{a.remote}/{a.branch}", commit)
                print(f"git_publish: pushed {commit[:10]} to {a.remote}/{a.branch}")
                new_head = commit
                break
            err = (proc.stdout + proc.stderr).decode(errors="replace")
            if any(s in err for s in ("non-fast-forward", "fetch first", "[rejected]", "stale info")):
                print(f"git_publish: {a.remote}/{a.branch} moved; rebuilding (attempt {attempt + 1})", flush=True)
                time.sleep(min(2 ** attempt, 10) * 0.2)
                continue
            raise PublishError(f"push failed:\n{err.strip()}")
        else:
            raise PublishError(f"{a.remote}/{a.branch} kept moving; gave up after {a.retries + 1} attempts")

        if a.no_advance:
            return EXIT_OK
        # Merged paths: the caller's file becomes the checkout's new content (their change plus upstream's).
        merged = [path for path, new, note in plan if note == "merged with upstream" and new is not None
                  and new[0] != "120000"]
        ok, msg = advance_checkout(git, new_head, write_back=merged)
        print(msg)
        return EXIT_OK if ok else EXIT_NOT_ADVANCED
    except PublishError as e:
        print(f"git_publish: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
