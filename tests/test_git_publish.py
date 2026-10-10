"""scripts/git_publish.py: publishing paths from a shared checkout without stashing or touching other sessions' work.

Each test builds a bare remote, a shared checkout ("dgx", where several sessions leave dirty and staged files) and a
second clone ("pop") that pushes concurrent upstream changes.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "git_publish.py"
SPEC = importlib.util.spec_from_file_location("git_publish", SCRIPT)
gp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gp)

ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("GIT_INDEX_FILE", raising=False)


def git(repo, *args, check=True):
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and proc.returncode:
        raise AssertionError(f"git {args}: {proc.stderr}")
    return proc.stdout.strip()


def write(repo, path, text):
    p = Path(repo) / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def read(repo, path):
    return (Path(repo) / path).read_text()


LINES = "".join(f"line {i}\n" for i in range(1, 21))


@pytest.fixture
def repos(tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "seed"
    git(tmp_path, "init", "-q", "-b", "main", str(seed))
    for name in ("mine.txt", "foreign_dirty.txt", "foreign_staged.txt", "other.txt"):
        write(seed, name, LINES)
    git(seed, "add", ".")
    git(seed, "commit", "-q", "-m", "seed")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-q", "origin", "main")
    dgx, pop = tmp_path / "dgx", tmp_path / "pop"
    git(tmp_path, "clone", "-q", str(remote), str(dgx))
    git(tmp_path, "clone", "-q", str(remote), str(pop))
    return remote, dgx, pop


def publish(repo, *args):
    return gp.main(["-C", str(repo), "--no-check", *args])


def foreign_work(dgx):
    """Other sessions' work in the shared checkout: a dirty file, a staged file, an untracked file."""
    write(dgx, "foreign_dirty.txt", LINES + "dirty, unstaged\n")
    write(dgx, "foreign_staged.txt", LINES + "staged\n")
    git(dgx, "add", "foreign_staged.txt")
    write(dgx, "foreign_untracked.txt", "untracked\n")


def assert_foreign_intact(dgx, also_dirty=()):
    assert read(dgx, "foreign_dirty.txt") == LINES + "dirty, unstaged\n"
    dirty = git(dgx, "diff", "--name-only").splitlines()
    assert dirty == sorted(["foreign_dirty.txt", *also_dirty])  # still dirty, not staged
    assert git(dgx, "diff", "--cached", "--name-only").splitlines() == ["foreign_staged.txt"]  # still staged
    assert git(dgx, "show", ":foreign_staged.txt") == (LINES + "staged").strip()
    assert read(dgx, "foreign_untracked.txt") == "untracked\n"
    assert git(dgx, "stash", "list") == ""


def remote_file(remote, path):
    return git(remote, "show", f"main:{path}")


def test_publishes_only_the_named_path_and_leaves_foreign_work_alone(repos):
    remote, dgx, _ = repos
    foreign_work(dgx)
    write(dgx, "mine.txt", LINES + "mine\n")
    assert publish(dgx, "-m", "Publish mine", "--trailer", "Co-Authored-By: X <x@example.com>", "mine.txt") == 0
    assert remote_file(remote, "mine.txt").endswith("mine")
    assert remote_file(remote, "foreign_dirty.txt") == LINES.strip()
    assert remote_file(remote, "foreign_staged.txt") == LINES.strip()
    msg = git(remote, "log", "-1", "--format=%B", "main")
    assert msg.startswith("Publish mine") and "Co-Authored-By: X <x@example.com>" in msg
    assert git(dgx, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")
    assert "mine.txt" not in git(dgx, "status", "--porcelain")  # published path is clean
    assert_foreign_intact(dgx)


def test_upstream_changes_elsewhere_are_kept_and_the_checkout_fast_forwards(repos):
    remote, dgx, pop = repos
    write(pop, "other.txt", LINES + "from pop\n")
    git(pop, "commit", "-qam", "pop change")
    git(pop, "push", "-q", "origin", "main")
    foreign_work(dgx)
    write(dgx, "mine.txt", LINES + "mine\n")
    assert publish(dgx, "-m", "Mine", "mine.txt") == 0
    assert remote_file(remote, "other.txt").endswith("from pop")
    assert git(remote, "log", "--format=%s", "main").splitlines()[:2] == ["Mine", "pop change"]
    assert read(dgx, "other.txt").endswith("from pop\n")  # clean path updated
    assert git(dgx, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")
    assert_foreign_intact(dgx)


def test_upstream_change_to_the_same_path_is_merged_three_way(repos):
    remote, dgx, pop = repos
    write(pop, "mine.txt", LINES.replace("line 2\n", "line 2 (pop)\n"))
    git(pop, "commit", "-qam", "pop edits mine.txt at the top")
    git(pop, "push", "-q", "origin", "main")
    foreign_work(dgx)
    write(dgx, "mine.txt", LINES.replace("line 19\n", "line 19 (dgx)\n"))
    assert publish(dgx, "-m", "Mine at the bottom", "mine.txt") == 0
    merged = LINES.replace("line 2\n", "line 2 (pop)\n").replace("line 19\n", "line 19 (dgx)\n")
    assert remote_file(remote, "mine.txt") == merged.strip()
    assert read(dgx, "mine.txt") == merged
    assert "mine.txt" not in git(dgx, "status", "--porcelain")
    assert_foreign_intact(dgx)


def test_a_conflict_aborts_without_pushing_or_touching_anything(repos):
    remote, dgx, pop = repos
    write(pop, "mine.txt", LINES.replace("line 10\n", "line 10 (pop)\n"))
    git(pop, "commit", "-qam", "pop")
    git(pop, "push", "-q", "origin", "main")
    before = git(remote, "rev-parse", "main")
    head = git(dgx, "rev-parse", "HEAD")
    foreign_work(dgx)
    mine = LINES.replace("line 10\n", "line 10 (dgx)\n")
    write(dgx, "mine.txt", mine)
    assert publish(dgx, "-m", "Conflicting", "mine.txt") == 1
    assert git(remote, "rev-parse", "main") == before
    assert git(dgx, "rev-parse", "HEAD") == head
    assert read(dgx, "mine.txt") == mine
    assert_foreign_intact(dgx, also_dirty=["mine.txt"])


def test_a_dirty_foreign_file_changed_upstream_blocks_only_the_local_advance(repos, capsys):
    remote, dgx, pop = repos
    write(pop, "foreign_dirty.txt", "pop rewrote it\n")
    git(pop, "commit", "-qam", "pop")
    git(pop, "push", "-q", "origin", "main")
    head = git(dgx, "rev-parse", "HEAD")
    foreign_work(dgx)
    write(dgx, "mine.txt", LINES + "mine\n")
    assert publish(dgx, "-m", "Mine", "mine.txt") == 3
    assert "foreign_dirty.txt (uncommitted changes)" in capsys.readouterr().out
    assert remote_file(remote, "mine.txt").endswith("mine")  # the push happened
    assert git(dgx, "rev-parse", "HEAD") == head  # the checkout did not move
    assert_foreign_intact(dgx, also_dirty=["mine.txt"])  # mine.txt now equals origin/main, HEAD is behind
    # Once the owner deals with the file, --sync-only advances, and the earlier published path becomes clean.
    write(dgx, "foreign_dirty.txt", "pop rewrote it\n")
    assert gp.main(["-C", str(dgx), "--sync-only"]) == 0
    assert git(dgx, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")
    assert git(dgx, "diff", "--name-only") == ""
    assert git(dgx, "diff", "--cached", "--name-only") == "foreign_staged.txt"


def test_dry_run_pushes_nothing(repos):
    remote, dgx, _ = repos
    before = git(remote, "rev-parse", "main")
    write(dgx, "mine.txt", LINES + "mine\n")
    assert publish(dgx, "--dry-run", "-m", "Dry", "mine.txt") == 0
    assert git(remote, "rev-parse", "main") == before
    assert git(dgx, "rev-parse", "HEAD") == before
    assert read(dgx, "mine.txt") == LINES + "mine\n"


def test_new_and_deleted_paths_and_directories(repos):
    remote, dgx, _ = repos
    foreign_work(dgx)
    write(dgx, "pkg/a.txt", "a\n")
    write(dgx, "pkg/b.txt", "b\n")
    os.remove(dgx / "other.txt")
    assert publish(dgx, "-m", "Add pkg, drop other", "pkg", "other.txt") == 0
    files = git(remote, "ls-tree", "-r", "--name-only", "main").splitlines()
    assert "pkg/a.txt" in files and "pkg/b.txt" in files and "other.txt" not in files
    assert "foreign_untracked.txt" not in files
    status = git(dgx, "status", "--porcelain")
    assert "pkg" not in status and "other.txt" not in status
    assert_foreign_intact(dgx)


def test_a_concurrent_push_is_retried_by_rebuilding(repos, monkeypatch):
    remote, dgx, pop = repos
    calls = []

    def race(attempt):
        calls.append(attempt)
        if attempt == 0:
            write(pop, "other.txt", LINES + "raced\n")
            git(pop, "commit", "-qam", "race")
            git(pop, "push", "-q", "origin", "main")

    monkeypatch.setattr(gp, "before_push", race)
    monkeypatch.setattr(gp.time, "sleep", lambda s: None)
    write(dgx, "mine.txt", LINES + "mine\n")
    assert publish(dgx, "-m", "Mine", "mine.txt") == 0
    assert calls == [0, 1]
    assert remote_file(remote, "other.txt").endswith("raced")
    assert remote_file(remote, "mine.txt").endswith("mine")
    assert git(remote, "log", "--format=%s", "main").splitlines()[:2] == ["Mine", "race"]
    assert git(dgx, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")


def test_nothing_to_publish(repos):
    remote, dgx, _ = repos
    before = git(remote, "rev-parse", "main")
    assert publish(dgx, "-m", "Nothing", "mine.txt") == 0
    assert git(remote, "rev-parse", "main") == before


def test_unpushed_local_commits_stay_local_and_the_checkout_merges_in_object_space(repos):
    remote, dgx, pop = repos
    write(pop, "other.txt", LINES + "from pop\n")
    git(pop, "commit", "-qam", "pop change")
    git(pop, "push", "-q", "origin", "main")
    write(dgx, "foreign_staged.txt", "another session's local commit\n")
    git(dgx, "commit", "-qam", "local, unpushed")
    local = git(dgx, "rev-parse", "HEAD")
    foreign_work_without_staged = LINES + "dirty, unstaged\n"
    write(dgx, "foreign_dirty.txt", foreign_work_without_staged)
    write(dgx, "mine.txt", LINES + "mine\n")
    assert publish(dgx, "-m", "Mine", "mine.txt") == 0
    # Only the named path is published, on top of the remote; the local commit is not.
    assert git(remote, "log", "--format=%s", "main").splitlines()[:2] == ["Mine", "pop change"]
    assert remote_file(remote, "foreign_staged.txt") == LINES.strip()
    # The checkout merged the remote into its local commit (no stash), keeping the local commit and dirty file.
    parents = git(dgx, "log", "-1", "--format=%P", "HEAD").split()
    assert parents == [local, git(remote, "rev-parse", "main")]
    assert read(dgx, "other.txt").endswith("from pop\n")
    assert read(dgx, "foreign_staged.txt") == "another session's local commit\n"
    assert git(dgx, "diff", "--name-only").splitlines() == ["foreign_dirty.txt"]
    assert git(dgx, "diff", "--cached", "--name-only") == ""
    assert git(dgx, "stash", "list") == ""
