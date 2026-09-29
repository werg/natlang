# Adaptation implementation evidence

Implementation of [the adaptation plan](../plans/ADAPTATION_SYSTEM_IMPLEMENTATION.md), verified on 2026-09-29. See [the author guide](ADAPTATION.md) for the installed SDK and CLI workflows.

| Plan gate | Implementation and verification |
| --- | --- |
| A. Contracts and vendor baseline | Strict artifact/run schemas, canonical identities, pinned Ax source hashes, license and dependency closure; artifact and upstream pure-operation tests. |
| B. Compiler inventory | Named and authored inline descriptors, labels, slots, captures, ownership and origin mapping; compiler and inventory tests. |
| C. Effective runtime view | Task-local overlays and source revisions; invocation, discovery, inspection and edits agree; concurrent bindings, stale callable references and cross-program recursion tested. |
| D. Program guidance | Separate system layer with imported/generated-child scope, depth and compaction behavior; baseline and scoped guidance tests. |
| E. Evaluation infrastructure | Fresh case workers, typed observations, independent local/final metrics, oracle extraction, judges, cancellation and explicit infrastructure failures; fixture and usage tests. |
| F. Native strategies | Ax-derived GEPA and reflection using native execution; seeded selection/frontiers, dependency groups, merges and acceptance tests. No Ax framework or Python execution path. |
| G. Durable runs | Atomic checkpoints, locks, caches, bounded history, immutable provenance and request ledgers; interruption/resume, crash and budget tests. |
| H. Public workflows | Installed SDK/CLI evaluation, search/resume, inspect, source export, revalidation, activation and rollback; package lifecycle tests. |
| I. Browser and examples | Portable artifact binding, Node/browser consumption, three packaged suites and separate optimizer entry points; Chromium smoke, browser declarations and package graph checks. |
| J. Live acceptance | Recorded local-model baselines, matched reflection/GEPA searches, all three guidance modes, repeated validation and saved-artifact Node/Chromium calls. Failures and missing measurements remain explicit. |

Final automated verification:

- `npm test`: 646 tests passed, none skipped; all 22 native conformance cases passed. Includes compilation, application builds and browser declaration checks.
- `node --test ts-host/test/package-distribution.test.mjs ts-host/test/package-adaptation-lifecycle.test.mjs`: all five tests passed, with no skips; installed package exports, notices, browser exclusions and actual CLI workflows against a local scripted model.
- `node scripts/check-staged-package.mjs npm-packages/node/dist` and the corresponding browser command: relative package imports resolve.
- Node and browser `npm pack --dry-run` checks passed. Node package contents include adaptation SDKs, recorded examples and vendor notices; browser contents omit evaluation and optimization modules.
- `node ts-host/scripts/adaptation-browser-fixture.mjs` and `npm --workspace @natlang/typescript-host run test:browser`: Node-produced artifact consumed in actual Chromium, including concurrent baseline/adapted tasks.
- `git diff --check`: clean.

## Live findings and limitations

The executor and reflection model were explicitly configured as local LFM2.5 350M base Q8. Searches used matched bounded allowances. These small hand-authored fixtures and two replicates measure lifecycle behavior and instability; they do not establish benchmark accuracy.

[The evidence guide](../examples/adaptation/README.md) describes the historical runs and their fixture changes. [The final compositional run](../examples/adaptation/evidence/acceptance-compositional-live-2026-09-29.json) explicitly invokes the named risk helper and authored inline decision helper and records independent local and final metrics. Its six searches across lambda-only, guidance-only and joint modes made 101 model requests and produced no promotable artifact. Twelve fresh baseline validation replicates were attempted: eight scored and four failed infrastructure checks. Scored cases covered all authored helpers; helper completion gates failed. Transport failures and elapsed-budget exhaustion are retained, rather than converted to semantic zeroes.

Search provenance and usage were recovered from durable run records. Failed validation replicate usage was not captured by that process version and remains unknown; scored replicates retain per-case usage. The current runner captures a gateway ledger for both successful and failed attempts. Unknown token counts and unavailable pricing remain null.

[Saved-artifact consumer evidence](../examples/adaptation/evidence/consumers-live-2026-09-29.json) records the same stateful artifact digest loaded by the packaged Node SDK and actual Chromium browser SDK. Both preserved the unrelated file and exposed the selected digest. Both scored zero: the Node call quiesced and the browser returned an incorrect answer. This verifies the deployment lifecycle, without demonstrating quality improvement.

Historical moderation artifacts precede the explicit-helper fixture change and are stale by design. Strict binding rejects them; adopting them requires explicit revalidation. No live run demonstrated an instruction-quality gain.
