# Vibestudio teacher collection to backbone SFT

The existing Vibestudio skill suites provide typed inputs, service fixtures and
independent outcome scorers. The teacher executes the actual program through the
natlang harness; the surrounding Vibestudio agent's answer is not a training target.

## Mechanism inventory

| Component | Role in this path |
| --- | --- |
| Vibestudio `scripts/natlang-eval.mjs` and teacher collection command | Host-owned source projection, suite fixture creation, teacher execution and exact-request replay. Own the collection manifest and episode provenance. |
| natlang `ts-host/src/evaluation/` | Isolated execution of each suite case, including inner executor requests/replies. Existing suite `score` remains the independent outcome oracle. |
| natlang `ts-host/src/teacher/` and `scripts/neuralese_data/teacher_gen.py` | Existing program-IR teacher collection; the suite bridge preserves fixture/scorer semantics rather than replacing them with handwritten task answers. |
| `scripts/neuralese_data/vibestudio_sft.py` | Verify collection bindings, split/source protection, exact replay evidence and full success; render admitted executor calls. |
| `scripts/neuralese_data/sft_sources.py` | Shared `TemplateRenderer`, held-v3/v2 closure and reasoning-free teacher prefix masking. Reused directly. |
| `scripts/neuralese_data/exp0_sft.py` | Shared `chains_of` prefix-sharing representation and the existing FP32-master/BF16 learner. No separate trainer. |
| Held eval, probes and collectors | No sampling law, scoring contract or live experiment changes. The next experiment pin consumes the built chains and judges held success. |
| Managed llama-server, wllama, Python native engine, SGLang actor | No serving field, scoring, sampling, cache or native-state change. `vibestudio_sft.py` calls the tokenizer-only `sft_sources.TemplateRenderer` (`render`, no model) and emits local learner data; the engines do not implement or consume the new collection-file schema. See `docs/ENGINE_PARITY.md` for the independent serving protocol. |

## Collection contract

The full schema is validated by `vibestudio_sft.inventory` and
`vibestudio_sft.episode_reason`. A `vibestudio.teacher-collection/1` manifest binds
the exact episode file (`episodes.path`, relative to the manifest, and
`episodes.sha256`) and lists **all** suite cases, including validation and test.
Each case carries `id`, `group`, `split`, `input_sha256`, `suite: {id, hash}` and
the three source-lineage arrays.
The manifest's top-level `suite`, `source` and `teacher` identities bind every
episode, including its program hash and model identity; a rehashed episode file
cannot silently mix another program revision or teacher into the collection.

A `vibestudio.teacher-episode/1` episode carries the matching case, suite and
lineage, the pinned clean template/natlang/Vibestudio commits, program SHA256 and
teacher identity. Each executor turn retains `messages`, `tools`, assistant
`target` and `invocation_id`. The independent outcome must be returned, quality 1,
all nonempty gates true and `tool_error_count: 0`. The initial admission is
conservative: tool-error recovery episodes are retained in collection but not
supervised until failed actions have explicit zero-weight attribution.
The separate semantic verification must carry `verdict: "accepted"` and a
nonempty evidence reason, and the episode must have no rejection.

Replay must execute the same fixture/program with recorded replies, independently
check the outcome again, and attest `verified: true`. Its `request_hashes` are the
ordered SHA256 digests of canonical compact JSON `{messages, tools}` for each
recorded call (object keys sorted recursively, UTF-8 Unicode). A flag alone does
not pass admission: hashes must equal the episode's actual requests.
These are trusted collector-code attestations and file-integrity checks, not
cryptographic signatures or a claim of proof against an adversarial producer.
When `turn.request_canonical` is present its parsed content must match the exact
messages/tools, and its UTF-8 bytes are hashed directly, preserving JS numeric
spellings that Python's JSON serializer would otherwise change.

Lineage is namespaced by template and suite:

```
source_ids:       ["vibestudio:TEMPLATE:SUITE:CASE"]
source_groups:    ["vibestudio:TEMPLATE:SUITE:GROUP"]
source_revisions: ["<exact template commit>"]
```

## Build and train

Run through the memory ledger on the DGX, with `NATLANG_REPO`/`PYTHONPATH` bound to
the next experiment's pin:

```
python scripts/neuralese_data/vibestudio_sft.py \
  --manifest COLLECTION/collection.json --episodes COLLECTION/episodes.jsonl \
  --tokenizer TOKENIZER --held-manifest HELD_V3/manifest.json --out BUILD
```

Use repeatable `--collection DIRECTORY` for collection directories containing
`manifest.json` to build all suites into one `chains.pkl`. Their inventories are
combined before admission, so protected source-group siblings in another
collection stay protected. `--manifest FILE` also derives its bound episode path
when `--episodes` is omitted. Mixing with another existing corpus is a next-pin
recipe decision, not a change to an in-flight learner.

Admission excludes every nontraining split, training groups shared with protected
splits, identical inputs crossing splits and held-v3/v2 ids/source groups/source
ids. Exact duplicate demonstrations and excessive attempts per case are removed.
Identical-input protection uses the stable template/suite source namespace across
suite build hashes, so a rebuild and renamed group cannot move a protected input
into training.
Malformed provenance, mutated inputs, missing split inventory or changed episode
files fail the build. Calls are admitted whole: an oversized call fails explicitly
instead of being truncated or silently dropping part of a successful trajectory.

`BUILD/chains.pkl` is directly consumable by `exp0_sft.py train --chains`.
`rows.jsonl.zst` retains semantic requests/replies, lineage and provenance for later
rerendering. `build.json` is the completion receipt, input hashes, tokenizer
identity, admission funnel and output hashes. The output directory is immutable:
choose a new path for a new build. Publish the actual collected data addition in
`training/neuralese_corpora.json` and through `sync_training_corpora.py publish`
before training; synthetic test fixtures are not a corpus addition. The build
retains exact collection manifests and episode bytes in `collections/` and checks
the source again after copying, so the published artifact contains its acquisition
evidence as well as admitted training rows.

Registration uses the existing shared publisher (including its check-main gate,
immutable manifests and preservation of unrelated staged work):

```
python scripts/neuralese_data/vibestudio_sft.py register --out BUILD \
  --register-id vibestudio-teacher-YYYYMMDD-v1 --register-owner dgx
```

The owner is explicit and must be the machine actually holding the data. A build
on a different host must be transferred to its owner before this command claims
DGX/Pop ownership. `--register-id ID --register-owner dgx` can also be appended to
the build command when building on the owner itself. External data gets a
gitignored `runs/ID` link through `register_corpora.register`, not a source-tree
copy. Registration verifies output and raw-input hashes again. Its registry entry
records collection, admission and rendering counts separately and explicitly says
that no learner exposure is performed by registration. No test fixture is ever
automatically published.

The assistant's own output carries loss. Every prompt, earlier history, service
observation and harness message is zero-weight context. An empty think block
inserted by the LFM3 template for a teacher with no visible reasoning is also
zero-weight, using the existing `sft_sources.reasoning_free_context` mechanism.
Teacher probabilities are neither required nor fabricated for SFT.

`build.json.target_mass` reports calls, target tokens and weighted target tokens
by suite/function/root-or-child and totals for root, child and unknown calls.
Function names, definition ids and parent invocation ids come from the actual
evaluation traces, and are retained in semantic rows. Missing traces are labelled
unknown; the report never guesses that a call is a root. This census measures the
actual trained mass of forwarding wrappers versus semantic child calls without
introducing any new weighting.

Targeted verification: `tests/neuralese/test_vibestudio_sft.py` checks outcome and
replay admission, source-group and identical-input split closure, held lineage,
provenance, deduplication, file immutability and the actual `exp0_sft` chain format
with zero-weight prompt and empty-think context. It does not establish learned
quality; that is the paired next-pin held eval and untouched Vibestudio scenarios.
