# Code capability corpus

Status: working acquisition, capture, native replay and app-dependency pilots, 2026-09-23.

## Objective

Teach the small interpreter to compile natural-language instructions into short
TypeScript programs, including complete function bodies in a single `eval`.
Collect broadly, keep provenance and confidence, and specialize on native
trajectories afterwards. Noncommercial datasets, including Case2Code, are in scope.
Do not require every source to be a native trajectory before it can be useful.

## Sources and priority

| Source | Collection | Initial use | Extension |
| --- | --- | --- | --- |
| es-toolkit | pinned Git checkout; JSDoc, implementation, Vitest tests | pure TS functions | small helper closures, callbacks |
| Radashi | pinned checkout; JSDoc/docs and tests | pure TS functions | optional/default args, helpers |
| simple-statistics | pinned checkout; JSDoc and tests | numerical JS | helper dependencies, numeric tolerances |
| Exercism TS and JS | track checkout and problem-specifications | descriptions, exemplars and tests | stateful exercises |
| Remeda | pinned checkout | source inventory | curry-wrapper normalization |
| Ramda | pinned checkout | source inventory | curry wrappers and callback fixtures |
| Case2Code | OpenMOSS-Team/case2code-data (formerly fnlp) | preserve Python code, inputs, outputs, prompt | translated TS candidates, differential replay |
| Salesforce xLAM | Salesforce/xlam-function-calling-60k | schemas and expected calls -> typed positional call exercises | fixture-backed compositions |
| Tiny-Codes | nampdn-ai/tiny-codes | JS/TS instruction/code candidates | generated descriptions/tests and replay |
| CodeSearchNet JS / Magicoder / McEval-Instruct | bounded HF snapshots | direct JS/TS description/code | execution verification |
| Deno std / JavaScript Algorithms / d3-array | pinned checkouts | documented source and tests | runtime-specific adapters |
| 30 Seconds of Code | pinned markdown snippets | description/code | runnable case synthesis |
| Stack-Edu | HuggingFaceTB/stack-edu | later educational raw-code reservoir | resolve Software Heritage IDs, extract functions |
| SPoC | human pseudocode + C++ + tests | later algorithm translation | retain line/block alignment |
| CodeAct / Code-Feedback | published interactions | later task and repair seeds | regenerate native executions |
| CommitPackFT / BugsJS / MEnvData | edits, bugs, repository trajectories | later repair/task extraction | reconstruct minimal runnable contexts |

Primary links: https://github.com/toss/es-toolkit,
https://github.com/radashi-org/radashi,
https://github.com/simple-statistics/simple-statistics,
https://github.com/exercism/typescript, https://github.com/exercism/javascript,
https://github.com/remeda/remeda, https://github.com/ramda/ramda,
https://huggingface.co/datasets/OpenMOSS-Team/case2code-data,
https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k,
https://huggingface.co/datasets/nampdn-ai/tiny-codes.

## Durable interchange

Use JSONL `natlang.code_task/1` records, independent of tokenizer and model.

```
{ version, id, group_id, kind, language, instruction,
  source: { name, revision, path, license, upstream_id },
  function: { name, parameters: [{name, type?}], return_type?, body, source, imports, helpers },
  cases: [{ args: [...], expected: value, outcome: "return" }],
  verification: { status, reasons? }, raw? }
```

`kind` is `function`, `io_synthesis`, `tool_calls`, or `instruction`.
Only function records with portable, executed cases enter native replay. Other
records retain original representations in `raw`, without claiming successful
verification. Source hashes, upstream revision, and function identity must survive
all projections. Missing descriptions, dependencies, unsupported types and errors
are explicit inventory/rejection reasons rather than silently invented context.
For calls, preserve `tools` and `calls`; derive argument order from an explicit
adapter signature, not an assumption about JSON object ordering at execution time.

## Implementation stages

1. Shared schema, stable identities, streaming JSONL, source registry, bounded
   local-file import, source inventories and manifests.
2. Case2Code, xLAM and Tiny-Codes adapters. Do not evaluate Python repr strings in
   the importer. Preserve exceptions and unparsable values. Export Python tasks as
   translation requests; attach later TS candidates by source identity.
3. Parse documented JS/TS function declarations with the TypeScript AST. Keep the
   original source and docs. Erase implementation-only generics for executable
   views; specialize portable boundary types per invocation. Initially reject free
   runtime bindings, callbacks, methods, generators, ambient effects, and mutation.
4. Capture calls during upstream tests with a source transform and a small capture
   runtime. Snapshot arguments before execution; preserve return/error, post-state,
   test identity where available, and function identity. Instrument function bodies
   rather than only exported bindings so internal test calls can also be observed.
   Run tests only in an explicitly selected disposable checkout/container. Capture
   failure must not alter test behavior. Concurrency must not corrupt JSONL records.
5. Replay portable calls against a fresh canonical native host with a scripted
   policy: `eval` body (returning the value), then a done reply. Capture exact native requests, actions and
   results, verify the expected value and unchanged input state, and export the
   existing `messages/tools/target` training format. Never fabricate host feedback.
6. Add translation jobs (Case2Code/SPoC), helper dependency contexts, portable
   callback fixtures, and real diagnostic->repair data. Verify translated semantics
   across multiple cases, not just syntax. Exclude hidden test outputs from prompts.
7. Mix native verified, source-tested, and unverified code examples as distinct
   training views. Preserve labels so ablations can measure the value of each pool.

## Admission and scope

The implemented replay subset is functions over finite numbers, booleans,
strings, null and arrays; plain-record boundary projection remains pending.
Preserve absence vs explicit
undefined in raw capture; do not silently JSON-coerce unsupported values. Defaults
must execute from the original signature. Closures, identity/aliasing, symbols,
cycles, BigInt, NaN, infinities, Date, native objects and function arguments are
initially recorded as unsupported, not mislabeled as ordinary JSON.

Whole-body eval can reject otherwise valid TS due to scope compiler rules. Keep
those sources available for ordinary code training and report native rejection.
Pure functions with sibling helpers can later bundle a bounded transitive closure.
The native host is not a security sandbox. Use process timeouts/resource limits,
and require explicit execution intent; arbitrary source test suites need an OS or
container boundary when not trusted. Never invoke network APIs from xLAM merely
to verify a call example.

## Splits, quality and scale

Group by original function/problem across every input, paraphrase, translation and
slice. Detect identical implementations across libraries before final splitting.
Keep upstream test splits held out. Cap invocation counts per function so heavily
tested functions do not dominate. Separate successful returns, expected throws,
and failures. Source tests establish observed behavior, not universal correctness.
Incomplete prose is acceptable; fabricated executions and incorrect associations
are not. Report counts by source, unique function, case, emitted turn, rejection,
input/output length, and verification tier. Pin revisions and hash local inputs.

Imports and replay must be bounded and streaming. Write output atomically and
refuse overwriting by default. Future resumable jobs key on source and adapter
digests. Large downloads and paid teacher runs are separate commands; a pilot
should require neither a GPU nor a model server.

## Validation and milestones

- Unit tests: schema/identity, streaming limits, malformed source rows, argument
  order, absent vs null, Python repr preservation, AST extraction, capture fidelity.
- Integration: run an actual test file with instrumentation, capture multiple
  invocations, replay through native host, and confirm genuine exportable turns.
- Reject changed outputs, mutation, unavailable dependencies, and unsupported
  cases. Verify all invocations remain in the same group and source provenance.
- Real pilot: pinned es-toolkit/Radashi and Exercism examples plus public dataset
  samples. Publish counts and exact reproduction commands, not estimated yield.
- Scale after pilot: coverage across sources; compare native-only, mixed-code,
  code-then-native training under comparable token budgets. Measure execution and
  program success, repairs, latency/tokens, and semantic-leaf retention. Hold out
  entire task families/compositions in addition to function identities.

## Current verified status

- Node build passes; 73 focused corpus, workspace, scope-compiler and native
  materializer tests pass, including type-only imports and sibling helper replay.
  Broader native-host/runtime checks still fail in the concurrently edited
  migration worktree; the latest check also caught an in-progress syntax error in
  native-host.test.mjs. No clean-baseline comparison was made. Full repository
  green status is not claimed.
- Seven-source snapshot is durable under `data/direct-code-2026-09-23` (gitignored).
  `bundle-final` contains 3,219 syntax-checked, deduplicated rows: 3,015 train and
  204 holdout. These are unverified code examples, not executed native trajectories.
  Source counts: CodeSearchNet 684, Magicoder 51, McEval 330, Deno std 975,
  JavaScript Algorithms 144, d3-array 56, 30 Seconds of Code 979.
- Extended acquisition under `data/extended-code-2026-09-23` adds Exercism TS
  (142 inventory rows), Exercism JS (239), Radashi (131), simple-statistics (104),
  Remeda (44), Ramda (17), and 1,000 Case2Code translation requests.
- After user-approved Hugging Face access was configured, both gated sources
  downloaded successfully: 1,000 raw rows each under `data/gated-code-2026-09-23`.
  xLAM contributes 962 schema-ordered TypeScript call examples; Tiny-Codes adds
  114 syntax-checked JS/TS examples after single-fence extraction. Original
  responses are preserved. Neither source is execution-verified.
  The latest joint bundle is `data/combined-code-with-gated-2026-09-23`:
  **4,968 examples, 4,664 train and 304 holdout**. All 22 dataset/source tests pass
  after the Tiny-Codes normalization fix. Credentials are not in corpus artifacts.
- Joint assembly at `data/combined-code-2026-09-23` deduplicates all accessible
  direct sources together before splitting: **3,892 examples, 3,654 train and
  238 holdout**. Python/translation rows are excluded. Source representative counts
  can differ from separately assembled bundles because deduplication is joint.
- Guarded `throw` bodies are now supported without stripping guards; runtime
  failures remain failures. Directly returned empty arrays use declared result
  types. The durable chunk pilot now admits 20/20 distinct replay cases (40 turns),
  and the d3 ascending pilot admits 8 cases (16 turns). The dependency-bearing
  transpose pilot passes its original suite (22 portable captures) and all 7
  replay cases (14 turns): **70 verified turns across three functions** in total.
  Artifacts: `data/direct-code-2026-09-23/d3-transpose-pilot-final`.
  The mean pilot passes its source suite but yields no portable captures because
  of optional undefined/callback arguments; it contributes no positive rows.
  This is pilot coverage, not proof of repository-wide compatibility.
- Code imports packages the way a module in the app workspace would; the app
  installs its own dependencies. Imported handles do not enter portable scope.
  Real npm-package example runs in handwritten and NL modes.
  See `docs/application-packages.md` for the contract and limitations.
- Dependency-bearing corpus replay accepts `--workspace`, rebases captured
  imports, and records manifest/lockfile hashes before/after plus observed host
  events. Tests verify relative modules and an installed local package. An exact
  observed return match does not imply repeatable or completely logged effects.
  A bounded transitive set of sibling function declarations is now captured and
  replayed block-locally; arbitrary module initializers/state remain unsupported.
- The final bundle is template-neutral. The earlier `train.sft.jsonl` is only a
  rendering smoke test against an unconfirmed server model template. That server
  is unavailable; final export needs the selected student model's template.
  No model training has been launched.

## Earlier milestones (historical; superseded where noted above)

- Implemented `ts-host/scripts/code-corpus/`: source acquisition/registry,
  generic and Exercism inventory, Case2Code/xLAM/Tiny-Codes imports, AST extraction,
  synchronous test instrumentation, portable snapshots, code/translation views,
  isolated native replay and existing-materializer integration. See that folder's
  README for commands and exact supported subsets.
- Native build passes. 32 focused and adjacent native materializer/export tests
  pass, including real eval/mark-lines fixture replay, nested arrays, wrong-output
  rejection, recursion, mutation detection, unsupported snapshots, timeout,
  source-level groups, and schema variants. Full repository test suite not run.
- Real Case2Code datasets-server pilot: 3 rows downloaded, 3 normalized unverified
  IO-synthesis records, 3 translation requests. Input SHA-256:
  `2f1464881f046f4fd4d54aabba661b3860312c77b0625117c4d3f7226e8f5ce7`.
  Artifacts: `/tmp/natlang-case2code-{sample,tasks,translations}-2026-09-23.jsonl`.
  The source is Python: do not mix its Python completions into the JS/TS training
  view. Python-to-TS generation and differential verification are not implemented.
- Actual es-toolkit upstream tests at commit
  `ee72fc74b763d8cb48095e1981a5bcef19ba5cee`: `chunk` passed 5 tests and captured
  108 portable calls (including expected throws); `compact` passed 1 test and
  captured 3 nonportable calls involving NaN/BigInt. Pilot keeps source/spec hashes,
  dependency lockfile, test report, source records and captures. Outputs:
  `/tmp/natlang-es-toolkit-chunk-pilot-2026-09-23` and
  `/tmp/natlang-es-toolkit-compact-pilot-2026-09-23-v2`.
- Important measured compatibility finding: 20 distinct successful `chunk`
  invocations replayed but **zero were admitted**. Its entire body contains
  `throw new Error(...)`, which the current native eval compiler rejects even
  when that branch is not taken. This is not a capture failure. Failed rows remain
  diagnostic, not positive training examples. A first attempt also exposed native
  array type syntax (`T[]`, not `Array<T>`); projection fixed and regression-tested.

## Remaining implementation slices

1. Expand dependency-bearing real upstream pilots and preserve enough module
   context for regeneration. Add pnpm/Yarn workspace support without rewriting
   their lockfiles via npm. Host-process isolation remains a deployment concern.
2. Expand the real-test pilot to guard-free functions, then repository-wide Vitest
   and Exercism runners; preserve test identity and shard capture files per worker.
3. Improve dependency/builtin and local-vs-input-write analysis, object/record
   boundaries, empty-array declared types, arrows and callbacks. Static inventory
   is conservative and currently over-flags builtins/local property writes.
4. Acquire accessible full JS/TS datasets in bounded shards; gated sources require
   accepted user access. Add parquet streaming and resumable content-addressed jobs.
5. Add Case2Code safe literal conversion, translation execution in isolated
   environments, and differential checks for JS/Python semantic differences.
6. Cross-source implementation deduplication and fixed evaluation partitions are
   implemented for assembled code views. Render with the intended student
   template, then run controlled code/native mixture experiments. Native and
   general-code views also need joint group-aware splitting before mixing.
