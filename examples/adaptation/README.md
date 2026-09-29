# Adaptation examples

Three isolated TypeScript suites cover named and interpolated inline instructions, compositional local/final metrics, and directory state assertions. Every case has a distinct source family and an explicit split. These are small hand-authored acceptance fixtures, not accuracy benchmarks.

From an installed Node/CLI package:

```sh
natlang adapt inspect ./triage --json
natlang optimize ./triage/suite.ts --dry-run
natlang eval ./triage/suite.ts --baseline --split validation
natlang optimize ./triage/suite.ts --strategy reflection --out ./runs/reflection
natlang optimize ./triage/suite.ts --strategy gepa --out ./runs/gepa
natlang optimize resume ./runs/gepa --suite ./triage/suite.ts
```

The checked-in suites explicitly resolve `default` for executor and reflection. Configure a reflection profile and update the suites if you want a different reflection model. Keep evaluation inputs and expected values out of the callable folders. Use the context runtime so candidate selection propagates through the application.

Execution policy and budgets are bounded examples. Real local runs may fail gates or produce invalid reflection output; inspect reports before activation. No improvement is an acceptable result. Infrastructure failures require repair and resumption, not scoring them as incorrect answers.

Recorded evidence lives in `evidence/`. `live-2026-09-29.json` contains fresh validation baselines; `search-live-2026-09-29.json` contains matched one-proposal reflection/GEPA runs. `acceptance-live-2026-09-29.json` records the moderation lambda-only/guidance-only/joint comparison and two fresh baseline/finalist validation replicates. It also retains earlier infrastructure and fixture-serialization failures. `acceptance-followup-live-2026-09-29.json` records subsequent triage/stateful calls after fixing failed-outcome serialization. Missing or stale artifacts are not forced through compatibility checks. Low scores, uncovered helpers and invalid proposals remain in these records.

`acceptance-compositional-live-2026-09-29.json` reruns the moderation comparison
after adding explicit helper calls and independent helper observations. Its coverage
includes the root, named risk helper, and authored inline decision helper. Historical
moderation artifacts precede this source change and are expected to fail strict binding.
Use revalidation to issue a new artifact; do not activate a historical artifact by force.
The final run made 101 search requests across six mode/strategy combinations and
produced no promotable artifact. Eight of twelve validation replicates scored; four
failed infrastructure checks. Durable search ledgers are included. Failed replicate
usage was not captured by that process version and remains explicitly unknown.
`consumers-live-2026-09-29.json` records a saved stateful baseline selection loaded in
the packaged Node SDK and actual Chromium browser SDK. Both traced the artifact digest
and preserved the unrelated file; both scored zero, with a quiesced Node call and an
incorrect browser answer. Successful loading does not establish correct model behavior.

Run the scripts with an installed/staged Node package:

```sh
node --experimental-strip-types run-live.ts
node --experimental-strip-types run-search-live.ts
node --experimental-strip-types run-acceptance-live.ts
node run-live-consumers.mjs
```

The acceptance runner accepts `--suites=triage,stateful` and a distinct `--output=results.json` for a follow-up cycle. Search allowances match across strategies; actual request counts and reported tokens are retained separately. These tiny suites and two replicates expose lifecycle errors and instability, not population accuracy or statistically reliable improvement. The local executor/reflection profile here is LFM2.5 350M base Q8. Reported provider failures and unknown usage are evidence, never semantic zeroes.

The consumer script additionally requires `@natlang/browser`, `playwright-core`,
and Chromium (or `NATLANG_CHROMIUM` pointing to its executable). It serves the
installed browser assets and bridges model turns to the configured local Node
session; it does not run optimization inside the browser.
