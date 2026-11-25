"""Own successes render under the prompt they were sampled with (plans/semantic-fp/02 R1 "training interaction";
DECISIONS 2026-11-23 "own successes render as recorded").

``sft_sources replay`` re-runs each extracted success under the replaying checkout's runtime, and ``build`` rendered the
replay's record of every call. For the student's own episodes (``rl-own``, ``jolt-guided``) that record is the same
decisions under the *replaying* runtime's system prompt, tool list and call openings (the replay admits an own episode
only when every call matched exactly or with a reworded opening). In the ei-sources WORK of 2026-11-23, 5,833 of 10,540
replayed own successes (55%) had a system prompt that differed from the one their calls were sampled with, e.g. a
plain-text final answer the recorded prompt allowed, trained under a prompt saying "A reply without a tool call is not
an answer". A target trained under a prompt it was not sampled with teaches the model to act against its prompt.

So an own candidate renders its *recorded* turns (messages and tools as the actor saw them); the replay stays the
verification that the decisions still succeed under today's runtime. Teacher episodes keep the replayed rendering
(their prompts were another model's; filled reasoning is drawn against the replayed context).

The recorded turns live next to the replay as ``WORK/replay/<name>/recorded-turns.json.zst``, written by ``replay`` for
new episodes and by ``backfill`` (one pass over WORK's extractions) for replays made before this file existed.
``sft_sources build --own-render replayed`` restores the replayed rendering.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

FILE = 'recorded-turns.json.zst'
OWN_SOURCES = ('rl-own', 'jolt-guided')
_KEY = re.compile(r'^\{"key": "((?:[^"\\]|\\.)*)"')


def path(work, name):
    return Path(work) / 'replay' / name / FILE


def write(work, name, turns):
    """The recorded semantic turns of one episode (their exact-id captures dropped: rendering does not read them)."""
    import zstandard
    from natlang_neuralese.common.jsonio import utf8
    target = path(work, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps([{k: v for k, v in turn.items() if k != 'exact'} for turn in turns], ensure_ascii=False)
    tmp = target.with_suffix('.tmp')
    tmp.write_bytes(zstandard.ZstdCompressor(level=6).compress(utf8(data)))
    tmp.replace(target)


def read(work, name):
    import zstandard
    with open(path(work, name), 'rb') as raw:
        return json.load(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding='utf-8'))


def backfill(work, names_by_key, episode_files, lines_of):
    """Write the recorded turns of every key of ``names_by_key`` ({episode key: replay name}) that has none yet, in one
    pass over WORK's extraction files (only the lines of those keys are parsed). Returns (written, missing keys)."""
    wanted = {key: name for key, name in names_by_key.items() if not path(work, name).exists()}
    written = 0
    for file in episode_files:
        if not wanted:
            break
        for line in lines_of(file):
            match = _KEY.match(line)
            if not match:
                continue
            key = json.loads(f'"{match.group(1)}"')
            if key not in wanted:
                continue
            episode = json.loads(line)
            write(work, wanted.pop(key), episode['turns'])
            written += 1
    return written, sorted(wanted)
