# Gemini generation pool

Google limits vary by model and apply to the project, rather than to each API key.
Use independent verified free text model budgets; do not multiply capacity by
creating keys or assuming that aliases have separate quotas.

## Typed decisions

`scripts/run_gemini_decision_pool.py` shares the existing HTTP request construction,
gold-free prompt, strict typed answer validation, and response provenance from
`scripts/label_decision_cases.py`. The declared inventory is
`training/providers/gemini-free-text-pool.json`. Update the inventory only after
checking current free text pricing and actual account availability. Catalog
presence alone is insufficient. Retired 2.5 models returned unavailable errors
on this account and are excluded. Preview aliases conservatively share a group
with their stable model until independent quotas are demonstrated.

Load `GEMINI_API_KEY` privately from `~/.config/natlang/gemini.env`, then run:

```sh
python3 scripts/run_gemini_decision_pool.py \
  --cases runs/my-source/cases.jsonl \
  --out runs/my-google-pool/labels.jsonl \
  --state runs/free-provider-generation-20261009-v1/google-project-quota-state-v1.json \
  --workers 4 --wait
```

All Google pool owners on this machine/project must use that same state file.
It is locked for the lifetime of the process; a second owner refuses to start
unless `--wait-for-owner` explicitly queues it behind the current owner. The
queued source reads the persisted quota state only after acquiring ownership.
Freeze the source, inventory and both adapter files before queuing.
Reserve groups used by older standalone workers with `--exclude-model MODEL`.
The exclusion applies to all aliases in the group. Coordinate cross-machine
ownership rather than starting competing project pools on both machines.

Four HTTP calls may run concurrently, with at most one per quota group. Ready
groups rotate without waiting for slow models. Per-group pacing is persisted
before the call. HTTP 429, server errors, and network failures put only that
group on exponential backoff and return the case to the queue for another ready
group. Actual Retry-After and nested Google RetryInfo delays take precedence.
Daily quota exhaustion without a reset delay is held for review rather than
assigned a guessed reset time. Configuration/account errors disable the group
for review. All attempts remain in `.attempts.jsonl`; only completed responses
enter labels, with their actual model identity. Invalid typed responses remain
explicit errors, never normalized probability distributions.

The pool state is mutable operational state, not a training corpus. Manifests
and closed label/attempt files are immutable publication inputs. Schema-valid
answers still require semantic grading and IR conversion before admission.
Restarting the same output requires identical source/config/adapter identity;
after changing code or configuration, use a fresh output and carry forward
case IDs explicitly through source selection.

## Full trajectories

Existing native Gemini trajectory queues remain model-specific so real thought
signatures and model context are preserved. Resume a partial trajectory with its
original model and runtime; do not rotate its continuation into another model.
Assign fresh whole cases across ready model queues and carry their quota
observations into the shared project state. Do not double-book their model
groups in the typed pool.

## Current Pop launch

`provider-google-model-pool-mixed-decisions-v1` contains 256 additional inherited
train cases from 19 families. The pool uses four concurrent groups; the running
3.1 Flash Lite standalone batch is reserved. Observed exhausted 3.5/3.6/3.8 Flash
groups were seeded with their actual provider reset delays. The earlier 3.5 Flash
Lite broader batch closed with 128/128 structurally valid responses; semantic
review is separate and pending. All resulting data is held, with zero new-world
credit. Do not claim that inventory size means all models produced answers.

Sources: [Google rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
[Google pricing](https://ai.google.dev/gemini-api/docs/pricing).

Standalone HTTP label queues pause after an HTTP429 exhausts bounded retries,
leaving later source cases pending. Use provider RetryInfo to distinguish a
minute cooldown from a daily cap; do not automatically relaunch before it.
A retryable transport error can be requeued with its earlier attempt preserved.
Semantic-invalid labels remain errors rather than automatic relabel targets.

The additional verified RoboticsER2 group is included in the next frozen pool;
its standalone worker has now closed on a daily quota. The next queued source
contains1024 additional inherited train cases and all10 verified model IDs
across9 conservative quota groups. Both prior Lite workers are closed.
