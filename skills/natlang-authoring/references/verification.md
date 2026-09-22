# Verification and diagnosis

## Separate what each result establishes

| Evidence | Establishes | Does not establish |
|---|---|---|
| Load/type check | Source graph and signatures are accepted | The model follows the algorithm |
| Scripted interpreter run | Real runtime plumbing and selected tool sequences work | Semantic quality of a teacher/student |
| Exact oracle/effect assertions | Computation, IDs, counts, or requested operations match expectations | Interpretations outside the assertions |
| Live interpreter trajectory | A particular model executed a particular scenario | General reliability or cross-platform determinism |
| Independent semantic review | Evidence supports the evaluated meaning | Correctness for unseen scenarios |

Load the bundled review example with Python:

```python
from pathlib import Path
from natlang.host import load
root = load(Path("review/review.nl"), {
    "observations": ["The trial improved response times."],
    "criterion": "Evidence of improved response times",
})
```

This loads and binds; it does not execute inference. For a fixture run use a scripted `ToolAgent` decoder or session agent and label it as such. Crisp helpers can execute without a model. For a live run supply the selected decoder and record its actual identity; never use fixture answers to claim model success.

## Scenario design

Use scenarios tied to desired behavior rather than only a happy-path demo. Cover ambiguity, conflicting evidence, missing information, zero/one/many items, order-sensitive updates, and interruptions where applicable. Include a case requiring multiple inspected operations. For stateful systems verify actual receipts and committed state; a model's prose about a completed effect is insufficient.

For a host-backed `Dict<T>`, verify that an ordinary eager dictionary still
works, directory reads do not fetch leaves, a selected leaf is fetched once,
the leaf is checked against `T`, traversal cannot escape the provider root, and
the same value can be passed to a typed child. Test provider reconstruction
separately from portable runtime-state restoration.

For stochastic behavior pin source, model/template, seeds, sampling, and ordered inputs, and report how many trials were run. Keep model sampling randomness separate from game/world randomness. Classify failures: source/loader, schema/transport, runtime, continuation, host/effect, model semantics, environment capacity.

In a checkout, useful checks are `.venv/bin/python -m pytest -q tests/test_codebase.py tests/test_surface.py`, `npm --prefix ts-host run build`, and the applicable files under `ts-host/test/`. Run only tests relevant to the change plus required project gates. A Python/TS contract change usually needs a paired case in `ts-host/test/native-parity.test.mjs`.

## Diagnose before modifying policy

- Rejected `inputs` path: distinguish interpreter paths from artifact IDs; put literal arguments in `values`. Do not unwrap arbitrary malformed values to make the test pass.
- Correct return type, wrong answer: improve semantic criteria, algorithm, evidence access, or model; structural validation is working as intended.
- Growing prompt: inspect presented schemas and repeated data, not just source length. Compact representation before adding language size limits.
- Read-only error on a read: verify host parity; input immutability should reject mutation, not inspection.
- Repeated work after rollover: check durable locals, restored pending nodes, marks, notes, and effect observations. Do not restart effects merely because conversation history changed.
- Capacity failure: inspect actual loaded context per server slot, model limits, and simultaneous workloads. A configured total context may be divided across slots. Never infer ownership or queue position from aggregate metrics alone.
- Low throughput: measure inference, waiting, host I/O, and context reconstruction separately. Avoid repeatedly announcing speculative progress from a quiet process.

`local` validation feedback lets the model inspect rejected actions and repair; `caller` returns diagnostics to the caller. This is a policy decision, not a correctness bypass. Neither repairs an external effect nor rolls it back. Browser applications default to local feedback; lower-level hosts and Python ToolAgent default to caller at the inspected revision. For model studies explicitly record the policy. Existing local repair has its own missing-return nudge behavior; it is not a guarantee of infinite successful retries.

## Training handoff

Capture source revision, typed inputs, exact presented messages/tools, actions, observations, seeds, continuations, model identity/settings, engine authority, and actual effects. Retain failed examples with clear labels. Do not automatically admit a trace because it parses, reaches `done`, or passes structural checks. Independently evaluate semantics and verify provenance. Split related variants by source program/family to avoid train/test leakage.

Repository anchors: `tests/test_continuation.py`, `tests/test_reduction_trace.py`, `natlang/scenario.py` when present, `ts-host/src/native/scenario.ts`, `ts-host/studio/research/evaluation.mjs`, `PROGRAM_IR_PIPELINE.md`, and `TEACHER_SETUP.md`. Use current runner flags rather than copying historical model-specific launch commands.
