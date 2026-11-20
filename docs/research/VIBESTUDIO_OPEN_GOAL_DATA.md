# Open user goals for Vibestudio teacher data

An open goal is the user's request and the initial Vibestudio workspace. The
Natlang root receives the request as a single string. Installed capability
definitions and authorized service interfaces stay visible. The harness keeps
expected outcomes, target selection, source coordinates, and evaluation checks
private. The teacher supplies the actions that connect the initial state to the
goal.

The rollout for each case records the root and every observed child invocation,
including attempts that failed and the resulting state. Admission requires
independent readback of the actual world. A case added to the task catalog
contributes zero training examples until a qualifying trajectory is captured.

## Running the open-goal lane

From the Vibestudio checkout, list the cases with
`pnpm natlang:teacher --list --open-goals`. The managed installation check is
`pnpm natlang:teacher --preflight --open-goals`. A capture uses
`pnpm natlang:teacher --open-goals --case CASE --out NEW_DIRECTORY` with explicit
case IDs. It uses the configured teacher and the chosen workspace's real runtime.

In the Natlang checkout, ingest each captured training manifest with
`scripts/neuralese_data/sft_sources.py extract --work WORK --live-run MANIFEST`
using the Neuralese Python environment. Pass all captures for the extraction
together or use distinct `--out-name episodes-NAME` values. Extraction stores
the verified open-goal call tree and model decisions as
`natlang.teacher_trajectory.native/1`. The SFT build derives its targets from
that IR after comparing it with the original captured records. Rejected
actions, unexecuted proposals, unsuccessful invocations, and score calls
remain evidence. Only eligible generation decisions carry supervised weight.

Pass `--live-trajectory-out teacher.native.jsonl.zst` at extraction to export
standalone IR. Decompress it to JSONL and pass it to the Natlang TypeScript
host's `scripts/materialize-native-teacher.mjs` with the original manifest and
its bound artifacts in place. The materializer checks their digests and
projects standard `natlang.teacher_training_turn.native/1` targets.

Save `pnpm natlang:teacher --list --open-goals` from the pinned Vibestudio
checkouts as a JSONL catalog. Each open goal declares its split and source
groups before capture. Pass the catalog to SFT with
`--live-holdout-registry CATALOG.jsonl`. The learner admits open-goal captures
whose identity and groups match a training entry and excludes the held
cases and their source-group siblings. Held variants stay out of training
extraction.

Assemble the SFT mixture with `sft_sources.py build`, passing
`--min-open-goal-share 0.8` to
enforce the contract even when extraction admitted no live Vibestudio cases.
The build automatically applies the minimum when verified Vibestudio live data
is present, including when every verified capture came from a legacy assignment.
Use `--open-goals-only --no-default-teacher-pool` to assemble a goal-only mixture
without including the historical teacher pool.

The minimum applies both to distinct admitted goals divided by all
training trajectories and to the weighted target tokens from their
admitted chains divided by all weighted target tokens. Multiple attempts
of one goal count once in the first numerator. An open-goal episode whose
targets cannot all be rendered is excluded. Historical assignments and
auxiliary data are included in the denominator when selected.

### One-case smoke on Pop

Use a clean host checkout and clean template checkouts. Install the host
dependencies in that checkout with `pnpm install --frozen-lockfile`. The
workspace packages need their own links when running from a detached
worktree. Select the checkouts with `VIBESTUDIO_TEMPLATE_CHECKOUTS` and
confirm the configured teacher with the managed readiness command.

Run one case at a time with the Natlang memory ledger. The collector starts
and stops a managed workspace for the selected case and uses the configured
model provider. On Pop, leave enough free host memory for the pinned held
evaluation before starting a smoke. Run `scripts/memory_ledger.py status`
and choose the host budget and reserve using that measurement.

```sh
python "$NATLANG_CHECKOUT/scripts/memory_ledger.py" run \
  --unit vibe-open-goal-smoke --budget-gb "$SMOKE_BUDGET_GB" \
  --reserve-gb "$SMOKE_RESERVE_GB" --no-resumable --workdir "$VIBESTUDIO_CHECKOUT" \
  --env "VIBESTUDIO_TEMPLATE_CHECKOUTS=$VIBESTUDIO_TEMPLATE_CHECKOUTS" \
  --env "VIBESTUDIO_MAX_CONCURRENT_BUILDS=1" \
  -- pnpm natlang:teacher --open-goals --case open-goal-reconcile-workshop-update \
  --out "$CAPTURE_DIRECTORY"
```

Wait for the unit to exit successfully and keep the entire capture directory.
The learner needs the actual tokenizer, chat template, and model type of the
starting checkpoint. Only tokenizer assets are loaded for data assembly.

```sh
python scripts/neuralese_data/sft_sources.py extract --work "$WORK" \
  --live-run "$CAPTURE_DIRECTORY/verified/manifest.json" \
  --live-trajectory-out "$WORK/smoke.native.jsonl.zst"
python scripts/neuralese_data/sft_sources.py build --work "$WORK" \
  --out "$LEARNER_READY" --tokenizer "$TOKENIZER_DIR" --base-rates '' \
  --open-goals-only --no-default-teacher-pool --min-open-goal-share 0.8 \
  --live-holdout-registry "$GOAL_CATALOG" --val-episodes 0
```

Check both open-goal shares in `build.json` and retain the corresponding
capture, IR, and learner chains. A one-case smoke verifies data flow.

## Capability coverage

The [pinned baseline](vibestudio-open-goal-baseline.jsonl) contains 60 skill
documents and 1,343 system-test scenarios from the host and all eight
template repositories. Its 1,082 pre-existing agentic tests span 151
capability labels and set a floor of **2,164 distinct open goals**. The five
goals from the new lane do not enlarge the baseline. The 253
`harness-review` cases remain inventory items until their model paths are
confirmed. The three TestKit suites are recorded separately.

A baseline case counts when it starts a direct agent, declares an agent
orchestration, submits model turns in an orchestrator, or is registered as a
managed teacher case. Candidate workspaces are derived from tracked source
paths and template dependencies. The live catalog remains authoritative for
installation. A nearby Natlang file similarly records a candidate source
relationship. Coverage requires a goal that actually exercises the capability
and independently verifies the outcome.

To reproduce the snapshot, check out the host and templates at the commits
recorded in the first row. In a disposable System Testing checkout, copy
`scripts/vibestudio/open_goal_catalog_probe.ts` from Natlang to
`skills/system-testing/catalog-baseline.test.ts`. Set
`VIBESTUDIO_TEMPLATE_CHECKOUTS` to the pinned template root and
`VIBESTUDIO_GOAL_CATALOG` to the desired output file. From Vibestudio run:

```sh
node scripts/run-userland-tests.mjs --template system-testing \
  --filter skills/system-testing/catalog-baseline.test.ts
pnpm natlang:teacher --list > "$TEACHER_REGISTRY"
```

From Natlang run:

```sh
python scripts/vibestudio/open_goal_inventory.py \
  --host "$VIBESTUDIO_CHECKOUT" --templates "$VIBESTUDIO_TEMPLATE_CHECKOUTS" \
  --catalog "$VIBESTUDIO_GOAL_CATALOG" --teacher-registry "$TEACHER_REGISTRY" \
  --out "$BASELINE_OUTPUT"
```

| Capability | Evidence in the first cases | Gap |
| --- | --- | --- |
| Discover relevant state | Agent chooses search queries and sources, or lists and reads workspace files | Browser and CDP, provider accounts, remote devices, and multi-workspace discovery |
| Choose and compose skills | Goal-only root chooses research, revision, or a revision that depends on the research result | Broad installed skill selection, branching and replanning across longer workflows |
| Execute independently verifiable work | Public evidence is read from the live cache. A workspace revision is independently read back | Publishing, deployment, integration changes, UI state, and durable cross-service outcomes |
| Recover safely | A concurrent file edit makes the agent reread, replan, and verify a revision that retains the competing edit | Service interruptions, partial commits, delegated work, and follow-up goals across sessions |
| Generalize to fresh goals | A source group ties related public tasks together | Real user-goal intake and held source groups for every capability family |
| Train from trajectories | The SFT importer and native materializer consume observed call trees in canonical trajectory IR | A live teacher smoke through the complete pipeline and measured results on held open goals |
