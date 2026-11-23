# Recovered Vibe Studio data handoff

Observed 2026-11-23 02:07 UTC. **Export, independent native materialization, registry registration, and archive publication are complete.** Extraction into the standing teacher work directory, student reasoning fills, and learner exposure are separate; the first two remain queued on the DGX. No new Luna trajectory was generated for these recoveries.

Durable archive: `/mnt/external/natlang-vibestudio-full-luna-campaign-yci2LB1N` on DGX. Every cohort below retains the complete original call graph, original failed outcomes, native evidence and selected original decisions.

| Dataset ID | Archive directory | Completed goals | Partial trajectories | Supervised decisions |
| --- | --- | ---: | ---: | ---: |
| `vibestudio-natlang-vibestudio-full-luna-campaign-yci2lb1n-recovered-independent-decisions-20261122` | `recovered-independent-decisions-20261122` | 0 | 10 | 42 |
| `vibestudio-natlang-vibestudio-full-luna-campaign-yci2lb1n-recovered-partial-outcomes-20261123` | `recovered-partial-outcomes-20261123` | 0 | 2 | 23 |
| `vibestudio-natlang-vibestudio-full-luna-campaign-yci2lb1n-recovered-completed-outcomes-20261123` | `recovered-completed-outcomes-20261123` | 2 | 0 | 40 |
| Total | | 2 | 12 | 105 |

Each directory contains `capture/*/manifest.json`, `receipt.json`, `trajectory.native.jsonl.zst`, `native-validation-output.jsonl`, and `exported.json`. The manifest and native IR distinguish partial evidence from completed user goals. The old `verified` marker field counts exported rows; `completed_goals`, `partial_trajectories`, `supervised_decisions` and `evidence_only` make its meaning explicit.

The two whole-outcome reassessments retain the original presentation and public-brief failures and add independently bound verdicts. Presentation validation checks the actual values, MDX, current revision and original native DOM receipt/source. Public-brief validation checks the original UTF8 source bytes/hashes and quote spans, plus a distinct typed Terra review of the full actual Luna application result. Its original reviewer program, input, provider replies and verdict are retained. The two new partials admit only exact native read results, independently reconstructed asset values and checked findings; wrong goal-level conclusions remain unsupervised.

## Executed provenance and checks

CPU export and independent Python → disk verifier → TypeScript → native CLI materialization ran on Tick with code `72e57a32a45f8cc4142881791f5f3ca04aaaf217`: unit `vibe-recovered-cpu-export-tick-v2-20261123`, 10.236 seconds, 736.9 MiB peak. All 14 rows and 105 eligible decisions materialized; zero rejected rows. A standard filesystem namespace preserved DGX's exact absolute artifact paths. Dataset owner/path remain DGX; original producer records truthfully name host `tick`, clean code `72e57a32a`, and the original journal records were transferred intact.

Standard registry promotion ran on Tick via `data_registry.py promote`, including its quick gate: unit `vibe-recovered-registry-promote-tick-v3-20261123`, 25.373 seconds, 1.1 GiB peak. Commit `eb045ac3cf59bce5c089d7b41339648e98f1d9ca` is pushed on `origin/main` and contains all three corpus entries and immutable manifests. Their admission is `registry-recorded`; this is not a learner-exposure claim.

Targeted checks passed: 63 loader tests; extraction/replay and held-reservation regression coverage; actual archived presentation/public-brief/native-read assessor positives and forged evidence negatives; native materialization of the real completed and partial captures. The full campaign's old failed raw captures remain retained. The merge repair also parsed all 1,195 tracked Python files without syntax errors.

## Live handoff and pending work

DGX code pin: `/mnt/external/natlang-vibestudio-full-luna-campaign-yci2LB1N/archive-code-72e57a32a45f`, created by ordinary git fetch/worktree. `archive-code-current`, `archive-code-revision-current`, and Tick's campaign `remote-code`/`remote-code-revision` point to it. The standing archive watcher `vibe-semantic-live-archive-watcher-20261123` is active and uses this pin. Future collector completion bookkeeping uses the newer `10836b152` driver; its export format is unchanged.

Administrative receipts in the archive:

- `tick-export.json`: actual per-cohort export and native materialization counts.
- `tick-export-registry/`: original isolated producer journal and immutable snapshots.
- `tick-registry-imported.json`: idempotent import receipt into DGX's live registry.
- `recovered-outcomes-20261123-published.json`: archive publication after the compatible fill-source gate.
- `recovered-export-plan.json`: exact current commands and phase details.
- `semantic-fill-consumer-config.json` / `semantic-fill-consumer-ready.json`: preserved filler arguments, endpoints and checkpoint identity; replacement source pin and submitted unit.
- `recovered-outcomes-20261123-ready.json`: written only after standing-work ingestion actually executes; its absence is not an export failure.

At the observation time the following jobs are **ledger-waiting, not executed**, with the required 15 GiB reserve. DGX ledger headroom was 3.3 GiB; no job from this strand held an active claim:

| Job | Purpose | Budget GiB |
| --- | --- | ---: |
| `vibe-native-materializer-build-72e57a32a45f` | Build the durable DGX native materializer for future local use | 3 |
| `semantic-next-round-guard-reassessment-20261123` | Enforce next-round source minimum while preserving checkpoint/optimizer/RNG identity | 0.3 |
| `colab-g4a-fill-semantic-1795399096152349224` | Compatible student reasoning fill consumer | 4 |
| `vsingest-semantic-poll-reassessment-20261123` | Standing archive-to-teacher-work ingester | 1.5 |
| `colab-g4a-fill-semantic-keep-reassessment-20261123` | Filler supervisor | 0.1 |
| `vibe-recovered-ingest-finalize-20261123` | Confirm cohort extraction and write the actual ingestion receipt | 1.5 |

The old incompatible filler was stopped before publication; its replacement preserves every argument except source code. Current SFT, actor, checkpoint, optimizer and RNG were untouched. The obsolete DGX producer was cancelled so Tick is the single producer of these exports. Ordinary archive retries respect the shared cohort `.export.lock` and committed marker. The superseded Tick 72e driver was removed with the official pin helper after checking for active process references; raw captures and produced training data remain intact.
