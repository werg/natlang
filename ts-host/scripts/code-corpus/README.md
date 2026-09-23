# Code corpus tools

**Eval rejects `var`.** Any code that becomes an eval body in training data (captured source functions, teacher repairs, synthetic implementations) must use `let`/`const`: rewrite each `var` declaration to `let` while preparing the data (the scoping differences are edge cases), rather than dropping the example.

## Latest integration

Seven additional sources now have runnable adapters: CodeSearchNet JS, Magicoder
JS/TS, McEval-Instruct JS/TS, Deno std, JavaScript Algorithms, d3-array and 30 Seconds
of Code. Commands below run from `ts-host/`:

```sh
node scripts/code-corpus/collect.mjs /absolute/new-snapshot-dir codesearchnet,magicoder,mceval,deno-std,javascript-algorithms,d3-array,30-seconds-of-code 1000
node scripts/code-corpus/rebuild.mjs /absolute/snapshot-dir /absolute/new-bundle-dir 1000
node scripts/code-corpus/render.mjs /absolute/new-bundle-dir/train.jsonl /absolute/train.sft.jsonl http://127.0.0.1:8081
```

The limit is a per-source input/inventory cap, not a promise of that many usable
JS/TS records. Collection preserves raw HF previews and pinned Git checkouts.
HF preview content hashes identify the exact rows; datasets-server revisions are
not claimed to be immutable repository snapshots. Rebuild needs no acquisition.
Assembly syntax-checks snippets, deduplicates AST-printed implementations, links
duplicate source groups before splitting, respects upstream holdouts and excludes
name-only descriptions. Its train/test rows include prompt/completion and chat
messages, and are **not execution-verified**.

Render uses the supplied server's template, not an assumed model template. Verify
that server is configured for the intended training model. The manifest records
the template hash. A teacher server may have the wrong template for a student.
No training run is launched by these scripts.

For gated Hugging Face datasets, provide an approved token to the process through
`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` using your local secret manager or shell
environment. The scripts send it only to HTTPS requests for `huggingface.co` and
`datasets-server.huggingface.co`; they do not print or persist the token. Retry
into a new output directory because collection destinations are create-only:

```sh
node scripts/code-corpus/collect.mjs /absolute/new-gated-snapshot-dir xlam,tiny-codes 1000
```

Accept each dataset's access terms on Hugging Face before retrying. The command
does not contain the token; the process inherits it from the environment.

Latest measured bundle: `data/direct-code-2026-09-23/bundle-final` at repository
root, 3,015 train and 204 holdout rows. The earlier `train.sft.jsonl` is a render
smoke test against the server available at that time, not the final student export.
That server subsequently became unavailable; render the final bundle against the
selected student template before training.

The expanded joint bundle is `data/combined-code-2026-09-23`: 3,654 train and 238
holdout examples, including the accessible Exercism and utility-library sources.
It is assembled from original task records together, not concatenated split files,
so cross-source duplicates are linked before assigning splits. The separate
`data/extended-code-2026-09-23/case2code.translation-requests.jsonl` contains 1,000
Python-to-TS requests, not translated solutions or verified JS/TS training targets.

After authorized gated acquisition, the latest joint bundle is
`data/combined-code-with-gated-2026-09-23`: **4,664 train and 304 holdout**.
The additional examples are 962 xLAM schema-ordered TypeScript calls and 114
Tiny-Codes JS/TS implementations, drawn from 1,000 raw rows of each source.
Tiny-Codes single JS/TS fences are extracted from prose while preserving raw
responses; ambiguous multiple fences remain subject to syntax rejection.
These additions are not execution-verified. No credentials are saved in artifacts.

Package imports and networking are documented in
[application-packages.md](../../../../docs/application-packages.md). These are
application capabilities, not reasons to discard dependency-bearing source code.
Replay accepts `--workspace /absolute/app-root` for dependency-bearing functions.
Package imports are preserved; relative imports of application functions
are converted into nested native codebase entries (the `foo.nl` / `foo/` layout)
only when explicit portable signatures permit it. Unsupported or recursive
subfunction graphs are rejected, never inlined into eval. Prepare upstream
dependencies explicitly before replay; imports do not install them. Replay records before/after
manifest and npm lockfile hashes, Node version, capabilities and observed host
events. Extracted functions also carry a conservative transitive set of sibling
function declarations; typed helpers are projected into child codebase entries.
Module-level variable initializers and
arbitrary closures are not captured. Passing means the observed return value matched the captured expectation,
not that external effects are deterministic or fully recorded. Imported code may
perform effects outside the observer. Repository-wide test-runner adapters and
general module-state/closure capture remain future work.

The earlier dependency-and-helper pilot is retained for provenance, but its
untyped helpers cannot be represented as native subfunctions and it is no longer
admitted by the production recipe. It can be rerun to inspect rejections with:

```sh
node scripts/code-corpus/direct-pilot.mjs --execute --function transpose --output /absolute/new-pilot-dir
```

For an already installed local project using Node's built-in test runner, capture
documented functions from one source file directly with the adapter:

```sh
node scripts/code-corpus/workspace-pilot.mjs --execute \
  --workspace /absolute/app-root --source src/math.ts --test test/math.test.ts \
  --function sum --function mean --output /absolute/new-workspace-pilot
```

The adapter requires `package.json`, an existing `node_modules`, and a test file
that runs with `node --test`. It instruments a disposable copy, runs that actual
test file with Node test isolation disabled so capture state is shared, then
replays the captured calls with the installed workspace and imports enabled.
The output contains selected `tasks.jsonl`, runtime `captures.jsonl`, replay
trajectories, materialized `.turns.jsonl`, a manifest with package/lock hashes,
and the source/test plus their relative import closure under `upstream/`.
Package dependencies are identified by the preserved package manifests and
lockfile hashes; node_modules itself is not copied. This adapter supports a
single selected source module and Node's built-in test runner; it does not claim
support for Jest, Vitest, arbitrary test commands, or source files outside that
module. Review trusted project code before opting into `--execute`; test capture
and native replay are not security sandboxes.

The training recipe accepts repeatable `--workspace-case` JSON specs. Each spec
selects one source/test pair and one or more functions from that source; use
multiple case files for additional source or test files. The recipe freezes the runtime first, runs capture stages before recipe
preparation, and adds each generated verified-turn file to preparation inputs.
For example, save this as `/absolute/cases/double.json`:

```json
{
  "workspace": "/absolute/app-root",
  "source": "src/math.ts",
  "test": "test/math.test.ts",
  "functions": ["double", "mean"],
  "instruction": "Return twice the input.",
  "license": "MIT"
}
```

For a single function, `"function": "double"` is also accepted. Then pass the
spec to recipe creation (repeat the flag for additional source/test pairs):

```sh
python scripts/create_training_pipeline.py --output /absolute/recipe.json \
  --workspace-case /absolute/cases/math.json
```

The workspace must be local, contain `package.json` and installed `node_modules`,
and the selected tests must run under Node's built-in test runner. Source and test
paths must stay inside the workspace. Capture records actual calls made by those
tests; the pipeline does not synthesize extra cases or claim more verified
examples than the replay outputs admit. Its verified counts are available in
each `captured-unit-tests/NNNN/manifest.json` and `.turns.jsonl` artifact.
The default coding recipe admits only captures compatible with the current native
subfunction layout. It verifies turns against manifests and excludes old captures
that used inline helpers or relative application imports. At the current retained
snapshot this leaves 120 execution-verified coding decisions across six source
functions before fresh synthetic generation. The recipe reports excluded capture
directories and reasons in `unit_test_corpus`.

The measured `d3-transpose-pilot-final` has 22 portable upstream-test captures,
7 accepted distinct native cases and 14 materialized turns. Combined with the
chunk and ascending pilots, that is 70 verified turns across three functions.
`--function mean` is also available as a negative compatibility pilot: its
undefined optional argument and callback captures currently prevent admission.

Implementation plan: [CODE_CORPUS.md](../../../../plans/CODE_CORPUS.md).
Run commands below from `ts-host/`. No model server or GPU is needed for collection
or deterministic native replay. Node 22+ and the existing package dependencies are
required. Build the native runtime first:

```sh
npm run build:node
node --test test/code-corpus-*.test.mjs
node scripts/code-corpus/sources.mjs list
```

## Collect and inventory repositories

```sh
node scripts/code-corpus/sources.mjs fetch es-toolkit /tmp/corpus-es-toolkit
node scripts/code-corpus/inventory.mjs /tmp/corpus-es-toolkit /tmp/es-toolkit-tasks.jsonl 1000
node scripts/code-corpus/sources.mjs fetch exercism-typescript /tmp/corpus-exercism-ts
node scripts/code-corpus/inventory.mjs /tmp/corpus-exercism-ts /tmp/exercism-tasks.jsonl 1000
```

Acquisition resolves a commit and records it in `.code-corpus-source.json`; use
`--revision COMMIT_OR_REF` to repeat a particular snapshot. Destinations must not
exist. Acquisition does not install dependencies or execute upstream scripts.
The registry also includes Radashi, simple-statistics, Remeda, Ramda, and secondary
code/repair datasets; **registry presence is not an implemented test adapter**.

Generic inventory currently extracts documented top-level named function
declarations. Arrow functions, classes, overload/curry wrappers, imported helper
closures, and cross-file type resolution need follow-up adapters. Exercism uses
declared exemplars (not solution stubs), associated descriptions and tests; its
exercise-level description may need function-specific alignment. JS and TS versions
of the same Exercism problem share a split group.

## Import downloaded datasets

```sh
node scripts/code-corpus/datasets.mjs --adapter case2code --input /data/case2code.jsonl --output /tmp/case2code-tasks.jsonl --limit 100
node scripts/code-corpus/datasets.mjs --adapter xlam --input /data/xlam.json --output /tmp/xlam-tasks.jsonl --limit 100
node scripts/code-corpus/datasets.mjs --adapter tiny-codes --input /data/tinycodes.jsonl --output /tmp/tinycodes-tasks.jsonl --limit 100
node scripts/code-corpus/views.mjs /tmp/case2code-tasks.jsonl /tmp/translation-requests.jsonl 100
```

Pass `--license` and `--revision` from the download's metadata. Import accepts JSONL
or a JSON array and records an input SHA-256 manifest. Parsing is row-bounded;
hashing still streams the whole input file. Existing outputs are not replaced by
default. Gated sources must be downloaded with your own approved access. The
`sources.mjs hf-rows` command is a small public preview utility, not a bulk downloader.

Case2Code Python and repr inputs/outputs are retained, not evaluated by the importer.
Its view is a **translation request**, with no invented TS answer. Translation and
differential verification are subsequent work. Tiny-Codes produces unverified code
SFT views; xLAM produces schema-ordered call views with explicit signatures, not
pretend API responses. Code views are a separate, template-neutral format, not
native tool trajectories and not yet rendered model-specific training text.

## Capture and replay

Reproduce the real-test pilots (downloads two pinned upstream files and installs
pinned test dependencies with lifecycle scripts disabled in a disposable directory):

```sh
node scripts/code-corpus/pilot.mjs --execute --output /tmp/new-chunk-pilot
node scripts/code-corpus/pilot.mjs --execute --function compact --output /tmp/new-compact-pilot
```

These validate capture, not automatic native compatibility. The original chunk
pilot exposed the runtime's ban on throw and missing empty-array context. Both
capabilities are now supported: the latest 20 selected chunk calls all replayed,
producing 40 native decisions. The D3 ascending pilot produced 8 accepted calls and
16 decisions. Compact's captured NaN/BigInt cases remain nonportable.

`extract.mjs extract|instrument --input FILE --output FILE --source NAME
--revision COMMIT --license LICENSE` exposes the single-file extractor/transform.
The instrumentation must be loaded with `capture-runtime.cjs` before test execution;
set `CODE_CORPUS_CAPTURE` to a fresh output path. Use the same source metadata for
extraction and instrumentation so function IDs match. Keep originals unchanged and
run the instrumented copy through the upstream test runner.

The capture runtime records **bound arguments after defaults**, successful returns,
throws, post-call inputs, and portability reasons. Recursion is supported. Unsupported
values are diagnostic-only sanitized data and marked `portable:false`; never replay
those sanitized values as if they were originals. Async/generators and destructured
parameters are currently skipped. Test identity, worker-sharded capture, strong
effect detection, and fully transparent instrumentation remain future work.

```sh
node scripts/code-corpus/replay.mjs --execute --input /tmp/tasks.jsonl --captures /tmp/capture.jsonl --output /tmp/native.jsonl --limit 100
```

Replay caps distinct invocations at 20 per function by default (`--cases N`). It
rejects conflicting outputs for identical inputs. This writes real native trajectory rows, `.rejected.jsonl` diagnostics, and
`.turns.jsonl` materialized training decisions. Only exact successful results are
admitted. Source/function groups and licenses survive materialization. Current
replay supports primitive/array boundaries, rejects observed mutation, and defers
empty-only arrays, object boundary types, callbacks and dependency linking. Types
are inferred across the function's observed cases, never specialized to literal
expected answers. Multiline descriptions become one instruction line for the
single-body eval projection; original descriptions remain in code-task records.

Replay workers have a ten-second wall-clock timeout. **They are not security
sandboxes.** `--execute` is for inspected/trusted source only; untrusted projects
require OS/container isolation and resource limits. Do not run xLAM network calls.

Materialized native turns work with the existing `scripts/export-native-sft.mjs`
renderer and its selected model template. Before bulk training, add cross-source
deduplication, source/problem holdouts, invocation caps and a mixture experiment.
Do not mix translation requests (null completion) into SFT examples.
