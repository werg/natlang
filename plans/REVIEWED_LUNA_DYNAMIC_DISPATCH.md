# Reviewed Luna dynamic dispatch

`scripts/run_reviewed_luna_dispatcher.py` is a scheduling layer for future
root-reviewed Luna campaigns. It replaces fixed case sharding when case
durations vary; it does not change collection, judging, admission, or retry
semantics. Every case still runs through `scripts/run_bonsai_queue.py` and the
frozen runtime named by the reviewed plan.

## Reviewed plan

The plan is immutable and invoked with its SHA-256:

```sh
python3 scripts/run_reviewed_luna_dispatcher.py PLAN.json --sha256 PLAN_SHA256
```

It must declare schema `natlang.reviewed_luna_dispatch_plan/1`,
`root_approved: true`, a unique `campaign_id`, the canonical checkout as `cwd`,
and the exact pinned dispatcher, queue supervisor, source, and frozen runtime receipt. Its
`cases` array is the complete approved master list; every row has one unique
source `index` and a deterministic integer `seed`. `slots` is one through five,
and provider/model/concurrency must be `openai-codex`, `gpt-6-luna`, and one.
The plan also names `campaign_root`, `claim_ledger`, `launch_record`, and the
generation `authority` file. `campaign_root` must be fresh below `runs/`.
Optional provider request configuration is also
hash-pinned. The plan's SHA binds all paths, limits, source indices, and runtime
identity before any claim is made.

The dispatcher fails closed if authority shows another live Luna campaign. This
keeps its slot count within the existing provider capacity while it owns the
master list; use an explicit reviewed handoff before starting a new campaign.

## Claim and execution behavior

One controller holds an exclusive campaign lock. It claims the next unclaimed
case, writes an fsynced one-case queue under that slot's campaign directory,
records its source/runtime/queue hashes in an append-only fsynced ledger, and
starts the existing queue runner. A slot claims again as soon as its previous
runner exits, so short cases do not leave a fixed shard blocking later work.
Each slot retains one journal across its cases. The queue runner reconstructs
failure streaks and provider retry deadlines from that journal, preserving the
existing exponential cooldown rather than resetting it per claim.

The dispatcher records a terminal ledger event only after finding the matching
queue-runner `finish` event. If a claim has no such finish when its process has
ended, it records an `abandoned` event and holds that index from automatic
retry. On resume, the exact same reviewed plan can continue unclaimed indices;
finished, failed, and abandoned claims stay attempted. A failed or abandoned
case needs a separately reviewed retry plan and campaign identity. A graceful
SIGTERM/SIGINT stops new claims, signals active queue runners so their normal
collector cleanup preserves partial evidence, and records the stop. Hard
controller loss is reconciled from its durable claim, queue, journal, and exact
runner argv before any remaining case is assigned.

The dispatcher registers only active `run_bonsai_queue.py` PIDs with
`generation_authority.reconcile_luna_authority`, keyed by the exact queue,
journal, runtime, source index, and campaign. Waiting controller slots are not
reported as live provider workers. It never edits a prepared master list or
reassigns a claim from another controller.

This mechanism creates collection evidence only. It does not mark generated
rows correct, qualify graph topology, admit data to a corpus, or authorize
training.
