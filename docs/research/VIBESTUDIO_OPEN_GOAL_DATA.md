# Open user goals for Vibestudio teacher data

An open goal is the user's request and the initial Vibestudio workspace. The
task selects an application-specific typed Natlang entry and named inputs.
Installed capability definitions, specialist lambdas and authorized service
interfaces stay visible. The harness keeps
expected outcomes, target selection, source coordinates, and evaluation checks
private. The teacher chooses actions and function calls that connect the initial
state to the goal. The native trajectory IR records that execution; a universal
final report is not part of the application contract. The dated sections below
retain historical release evidence; the semantic application migration is current.

The rollout for each case records the root and every observed child invocation,
including attempts that failed and the resulting state. Admission requires
independent readback of the actual world. A case added to the task catalog
contributes zero training examples until a qualifying trajectory is captured.
When the runtime records the child inventory, admission checks every
invocation against it. The native materializer verifies its digest against
the original call artifacts before projecting training targets.

## Running the open-goal lane

From the Vibestudio checkout, list the cases with
`pnpm --silent natlang:teacher --list`. The managed installation check is
`pnpm natlang:teacher --preflight`. A capture uses
`pnpm natlang:teacher --case CASE --out NEW_DIRECTORY` with explicit case IDs.
It uses the configured teacher and the chosen workspace's real runtime. Use
`--assignments` to select the historical assigned workflows.

For a Codex teacher, `--approve-codex-session` lets the managed capture
answer the model gateway's provider approval in its disposable workspace.
It covers the gateway's authenticated Codex credential for that session.
Other approval requests stay queued for their owners. Give longer workflows
an explicit wall-clock budget with `--test-timeout-ms`. Cases whose catalog
row has `requiresReview` need `--review-model PROVIDER:MODEL` with a model
distinct from the teacher to evaluate the authored content independently.

In the Natlang checkout, record each immutable capture directory in the data
registry, then ingest each captured training manifest with
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

Save the output of `pnpm --silent natlang:teacher --list` from the pinned Vibestudio
checkouts as a JSONL catalog. Each open goal declares its split and source
groups before capture. The catalog includes the hash of the exact Natlang root
input. Pass the catalog to SFT with
`--live-holdout-registry CATALOG.jsonl`. The learner admits open-goal captures
whose identity, groups, and goal hash match a training entry. The registry
rejects source groups and exact prompts that cross splits. Held cases and their
source-group siblings stay out of training
extraction.

Assemble the SFT mixture with `sft_sources.py build`, passing
`--min-open-goal-share 0.8` to
enforce the contract even when extraction admitted no live Vibestudio cases.
The build automatically applies the minimum when verified Vibestudio live data
is present, including when every verified capture came from a legacy assignment.
Use `--open-goals-only --no-default-teacher-pool` for the isolated smoke.
For a mixed build, ingest new and existing verified manifests together and
build with the selected sources. Each historical assignment retains its original
capture and supervision. It can contribute to the mix when both open-goal
shares pass. The build evaluates the shares before publishing its rows, so a
rejected mix leaves the previous admitted rows available.
The current importer accepted all five archived seed-v3 assignment captures,
retaining 124 supervised calls under their original task identities. A build
that uses all five episodes needs at least 20 distinct verified open goals
and at least four times the selected non-open weighted target mass.
The mixed smoke imported all six trajectories and rejected publication at
16.7% distinct open goals and 10.4% weighted open targets. The five archived
episodes remain registered as `vibestudio-live-workflow-seed-20261118-v3`.

The minimum applies both to distinct admitted goals divided by all
Vibestudio training trajectories and to the weighted target tokens from
their admitted chains divided by all Vibestudio weighted target tokens
(chains of `vibestudio:` tasks, `--open-goal-scope vibestudio`, the
default). Multiple attempts of one goal count once in the first
numerator. An open-goal episode whose targets cannot all be rendered is
excluded. Historical assignments, in the build or as auxiliary parts, are
in the denominator. A mixed expert-iteration build's other parts are not:
its open goals are held to the contract against the other Vibestudio data
only. `--open-goal-scope all` measures every chain, as before. Filled
reasoning chains that replace an open-goal episode carry its fingerprint.

The live Luna campaign enters expert iteration through
`scripts/neuralese_data/live_campaign_ingest.py poll`, which turns each
committed campaign directory (its `exported.json` written) into extractions
of one growing WORK. Pass that WORK to `expert_iteration.py async-drive
--live-work`. Each round links its extractions, builds against its catalog
copy (`live-holdout-registry.jsonl`), and fills its episodes first.
The chain bundle retains each admitted goal's fingerprint and the required
share. The learner checks the planned optimizer schedule after applying its
context masks and any step limit, before loading model weights.

### One-case smoke on Pop

The [2,324-goal release](vibestudio-open-goal-pop-pins-composed-2324-framing.tsv)
has verified detached compositions on Pop at
`/home/werg/vibestudio-worktrees/open-goal-composed-2324-framing.Q877LO`.
To recreate a release, set `RELEASE` to a fresh directory, `RELEASE_PINS`
to its TSV, and `CATALOG_SHA256` to the hash recorded below. Set
`LEDGER_CHECKOUT` to the Natlang checkout with the current ledger and
`STAGE_UNIT` to a fresh unit name for the Host dependency and compiler checks.

```bash
cd "$LEDGER_CHECKOUT"
python scripts/memory_ledger.py run --unit "$STAGE_UNIT" \
  --budget-gb 3.2 --host-max-gb 3.2 --reserve-gb 4 --hold-budget \
  --no-resumable --priority low --wait 4200 \
  -- bash scripts/vibestudio/stage_open_goal_release.sh \
    "$RELEASE_PINS" "$RELEASE" "$CATALOG_SHA256"
```

The script installs the release's workspace links, verifies the native Host
dependencies, builds infrastructure with one workspace job at a time, and
checks both the combined and System managed compositions. The resulting
`$RELEASE/staged` marker follows exact catalog and Natlang reviewer verification.
For a fresh release with a prepared Host on the same machine, pass
`env VIBESTUDIO_INFRASTRUCTURE_DONOR="$PREPARED_RELEASE/host"` between the
ledger's `--` and `bash`. Staging checks all infrastructure source and toolchain
digests, copies the generated files, and verifies their output hashes before
the Host checks. Omit the variable for a cold build or when inputs changed.

Set
`VIBESTUDIO_TEMPLATE_CHECKOUTS` to `$RELEASE/templates`,
`VIBESTUDIO_CHECKOUT` to `$RELEASE/host`, and `NATLANG_RUNTIME_CHECKOUT`
to `$RELEASE/natlang`. The last checkout supplies the release's vendored
runtime. Set `LEDGER_CHECKOUT` to the checkout with the current memory
ledger and choose a fresh `SMOKE_UNIT` for each attempt.
From the Host checkout, `node --import tsx scripts/check-open-goal-runtime.mjs`
checks linked Natlang type environments and explicitly named callable paths
using the installed Workerd runtime. Pass a template name such as `system`
to inspect that workspace, or `system+system-testing` for its managed test
composition.

Run one case at a time with the Natlang memory ledger. The collector starts
and stops a managed workspace for the selected case and uses the configured
model provider. On Pop, leave enough free host memory for the pinned held
evaluation before starting a smoke. Run `scripts/memory_ledger.py status`
and choose the host budget and reserve using that measurement.

```sh
export VIBESTUDIO_TEMPLATE_CHECKOUTS NATLANG_REPO="$NATLANG_RUNTIME_CHECKOUT"
cd "$VIBESTUDIO_CHECKOUT"
pnpm --silent natlang:teacher --list > "$GOAL_CATALOG"
python "$LEDGER_CHECKOUT/scripts/memory_ledger.py" run \
  --unit "$SMOKE_UNIT" --budget-gb "$SMOKE_BUDGET_GB" --host-max-gb "$SMOKE_BUDGET_GB" \
  --reserve-gb "$SMOKE_RESERVE_GB" --hold-budget --no-resumable --workdir "$VIBESTUDIO_CHECKOUT" \
  --env "VIBESTUDIO_TEMPLATE_CHECKOUTS=$VIBESTUDIO_TEMPLATE_CHECKOUTS" \
  --env "NATLANG_REPO=$NATLANG_RUNTIME_CHECKOUT" \
  --env "VIBESTUDIO_MAX_CONCURRENT_BUILDS=1" \
  -- pnpm natlang:teacher --case open-goal-reconcile-workshop-update \
  --approve-codex-session --test-timeout-ms 600000 --out "$CAPTURE_DIRECTORY"
```

Wait for the unit to exit successfully and keep the entire capture directory.
For assembly on a different machine, copy that directory and the exact JSONL
catalog from Pop. The manifest's bound validation, call, and readback artifacts
use paths relative to the manifest. Use a machine with the Neuralese Python
environment and point `CAPTURE_DIRECTORY` and `GOAL_CATALOG` at the transferred
copies. Activate that Python environment before running the commands below.
Set `NATLANG_CHECKOUT` on the assembly machine to a separate Natlang data
checkout at `b41e062e9` or a descendant. This version provides capture directory
import and checks the open-goal share in the learner's actual step schedule.
The learner needs the tokenizer and chat template from the starting model.
Only tokenizer assets are loaded for assembly. Use that machine's held-v3
manifest for `HELD_MANIFEST`, with the referenced held-v2 exclusion available.
Place the capture, work, and learner output directories under a configured
logical data root. Use fresh directories and set a unique `SMOKE_DATASET_ID`
for this capture.

```sh
cd "$NATLANG_CHECKOUT"
export NATLANG_REPO="$NATLANG_CHECKOUT"
python scripts/data_registry.py record --id "$SMOKE_DATASET_ID" \
  --path "$CAPTURE_DIRECTORY" --kind teacher-demos \
  --format 'natlang.live-teacher-task/1 with bound Vibestudio evidence'
python scripts/neuralese_data/sft_sources.py extract --work "$WORK" \
  --live-run "$CAPTURE_DIRECTORY/verified/manifest.json" \
  --live-trajectory-out "$WORK/smoke.native.jsonl.zst"
python scripts/neuralese_data/sft_sources.py build --work "$WORK" \
  --out "$LEARNER_READY" --tokenizer "$TOKENIZER_DIR" --base-rates '' \
  --held-manifest "$HELD_MANIFEST" \
  --open-goals-only --no-default-teacher-pool --min-open-goal-share 0.8 \
  --think-treatment none \
  --live-holdout-registry "$GOAL_CATALOG" --val-episodes 0
```

Check both open-goal shares in `build.json` and retain the corresponding
capture, IR, and learner chains. A one-case smoke verifies data flow.

### Bounded multi-case capture

Generate a capture list against the exact release catalog and the managed
reference evidence. Add `--family` to choose a verified family. Use
`--local-train-cases-out` for workspaces that lack attached devices, an
Android emulator, deployment packages, public URL access, or a search provider.
The audit lists the prerequisites for each goal. Use `--local-only` when
running the corresponding model-free reference sweep on that host.

```bash
cd "$NATLANG_CHECKOUT"
python scripts/vibestudio/audit_open_goal_references.py "$GOAL_CATALOG" \
  "$MANAGED_REFERENCES" --train-cases-out "$TRAIN_CASES"
mapfile -t scenarios < "$TRAIN_CASES"
case_args=()
for scenario in "${scenarios[@]}"; do case_args+=(--case "$scenario"); done
```

Pass `"${case_args[@]}"` to the ledgered collector command above in place of
the single `--case` selector and give the capture a fresh output directory.
The collector runs cases serially within each workspace and baseline. On
releases that support `--cases-per-instance`, the collector starts a fresh
instance after 12 cases by default. A lower value trades more startups for
shorter-lived workspaces.

Register each complete capture directory and pass it to extraction with
`--live-capture-root "$CAPTURE_DIRECTORY"`. The importer discovers the
verified manifests and validates each before converting it to trajectory IR.

To continue an interrupted batch, plan its pending selectors against the same
catalog and write the next capture to a fresh directory. The planner uses the
same manifest verifier as the training importer. It checks the captured goal
text, family, and source groups against the current catalog before reusing it.
Pass `--capture-root` once for each earlier batch. Keep successful artifacts
from every batch as inputs to extraction.

```bash
python scripts/vibestudio/pending_open_goal_cases.py "$GOAL_CATALOG" "$TRAIN_CASES" \
  --capture-root "$PRIOR_CAPTURE_DIRECTORY" > "$PENDING_CASES"
```
A case that fails leaves its logs in its own directory. Retry it with a
fresh output directory and pass both roots in the same extraction.

### Feeding the assembled data into the learner

Use the registered `chains.pkl` produced by the build. On the DGX, set
`LEARNER_PIN` to the detached Natlang experiment checkout,
`NEURALESE_PYTHON` to the Neuralese Python interpreter, `LFM3_HF_BASE` to
the model and tokenizer directory used to build the bundle, and
`LFM3_NATIVE_START` to the matching native starting checkpoint. Choose a
fresh run output directory and size the ledger claim for the planned batches.

```bash
python "$LEDGER_CHECKOUT/scripts/memory_ledger.py" run \
  --unit "$TRAIN_UNIT" --priority high --hold-budget --no-resumable \
  --budget-gb "$TRAIN_BUDGET_GB" --reserve-gb 15 --workdir "$LEARNER_PIN" \
  --env "NATLANG_REPO=$LEARNER_PIN" \
  --env "PYTHONPATH=$LEARNER_PIN/training/neuralese:$LEARNER_PIN" \
  -- "$NEURALESE_PYTHON" "$LEARNER_PIN/scripts/neuralese_data/exp0_sft.py" train \
  --chains "$LEARNER_READY/chains.pkl" --base "$LFM3_HF_BASE" \
  --checkpoint "$LFM3_NATIVE_START" --out-dir "$TRAIN_OUT" \
  --epochs "$TRAIN_EPOCHS" --lr "$TRAIN_LR"
```

The learner emits `planned_open_user_goals` for the actual schedule before
loading model weights. The build's `open_user_goal_depths` and the learner's
`planned_open_user_goal_depths` show how much supervised weight goes to
root and nested invocations. Depth 0 is the root. One goal can contribute
at several depths. Compare the trained checkpoint with its starting
checkpoint on the same pinned held evaluation. Keep the Vibestudio held
goals and their source groups outside the training bundle.

The expert-iteration driver also accepts the registered bundle through
`--aux "vibestudio=$LEARNER_READY/chains.pkl,share=all"`. Supply those
trajectories once in the mix. Read `build.json` under `mix` for the
effective weights and open-goal shares, including auxiliary bundles.
The same mixture checks apply before the learner starts.

## Capability coverage

The [pinned baseline](vibestudio-open-goal-baseline.jsonl) contains 60 skill
documents and 1,343 system-test scenarios from the host and all eight
template repositories. Its 1,082 pre-existing agentic tests span 151
capability labels and set a floor of **2,164 distinct open goals**. The five
goals from the new lane do not enlarge the baseline. The 253
`harness-review` cases remain inventory items until their model paths are
confirmed. The three TestKit suites are recorded separately.
The audit reports evidence for all inventory labels. The 151 labels on
the pinned agentic tests determine agentic capability coverage.

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

From the clean Vibestudio checkout, use the installed-skill probe under the
memory ledger to reconcile the six workspace skill catalogs with the pinned
sources. It starts one managed workspace at a time and reads its actual
skill list and documents through the agent CLI. The report records Base and
System Testing as composition layers.

```sh
node --import tsx "$NATLANG_CHECKOUT/scripts/vibestudio/open_goal_installed_skills.mjs" \
  --host "$VIBESTUDIO_CHECKOUT" --templates "$VIBESTUDIO_TEMPLATE_CHECKOUTS" \
  --baseline "$NATLANG_CHECKOUT/docs/research/vibestudio-open-goal-baseline.jsonl" \
  --out "$SKILL_INVENTORY"
```

Use `--plan` to inspect the predicted catalog without starting a workspace.
For a one-workspace smoke, pass `--workspace examples`. The report sets
`complete` only when the live catalogs of all six templates match their
expected source owners, document digests, and visible metadata.

The pinned managed inventory matched 254 visible entries across all six
workspaces, representing 64 distinct source documents. The snapshot records
the exact revisions and original report digest. Re-run the probe after
adding skills. [Pinned managed inventory](vibestudio-installed-skill-snapshot.json)

| Capability | Evidence in the first cases | Gap |
| --- | --- | --- |
| Discover relevant state | The agent chooses search queries and sources, discovers workspace files, and inspects the live DOM and visible controls in an owned panel | Provider accounts, remote devices, and discovery across panels and workspaces |
| Choose and compose skills | Goal-only root chooses research, revision, or a revision that depends on the research result | Broad installed skill selection, branching and replanning across longer workflows |
| Execute independently verifiable work | Public evidence is read from the live cache. Workspace revisions and ordered browser changes have independent readback. Declared suite execution checks the current workspace receipt | Broader CLI workflows, publishing, deployment, integration changes, and durable cross-service outcomes |
| Recover safely | A concurrent file edit makes the agent reread, replan, and verify a revision that retains the competing edit | Service interruptions, partial commits, delegated work, and follow-up goals across sessions |
| Generalize to fresh goals | A source group ties related public tasks together | Real user-goal intake and held source groups for every capability family |
| Train from trajectories | The SFT importer and native materializer consume observed call trees in canonical trajectory IR | A live teacher smoke through the complete pipeline and measured results on held open goals |

The browser lane has four training goals and one held-out goal on distinct
source groups. It includes read-only inspection and a dispatch task where the
order of two changes matters. Each activation uses the latest observed
revision and retains the CDP renderer receipt. The verifier checks which rows
changed, the sequence of actions, the final DOM, and the unchanged workspace
source. The System Testing qualification selector
`open-goal-oracle-sequence-studio-dispatch` exercises the actual panel and
verifier with a reference sequence.

The suite lane adds five training goals and one source-separated holdout.
The open-goal program discovers actual runtime declarations, decides among
the observed alternatives, and delegates to the existing Natlang suite
review tree. It distinguishes planning, an authorized focused run, and a
filter that matched no tests. The verifier re-resolves the declaration in
the owned context and checks the exact execution receipt. Reference selectors
`open-goal-oracle-verify-adventure-world-state` and
`open-goal-oracle-verify-adapter-selected-module` cover Workerd and native
execution respectively.

### Workspace template cases

The pinned 343-goal release declares 268 training goals and 75
source-separated holdouts across six workspace templates. The same Base entry point
receives the goal across these workspaces. It links the template's installed
Natlang subtree and scopes the template services to the functions that use
them. The workspace and the reference checks are selected by the harness.

| Workspace | Goals | Baseline capability and open-goal coverage |
| --- | ---: | --- |
| Examples | 110 | Public evidence and document recovery, ordered CDP changes and readback, visual recovery across renderer revisions, declared suite verification, publishing, native PDF evidence, Natlang program execution and signing, and native eval recovery |
| Personal | 84 | Declared tests, Mission Control, browser-library-to-collection work, browser migration, archived decisions, and coordination across authorized monitors. Exact readback checks recovered references, historical authority, and interrupted scheduling changes. |
| System | 55 | Declared tests, hub and terminal operations, retained conversations and cursor recovery, Host diagnostics, and owned local delivery with interrupted acknowledgement recovery |
| News | 12 | `news-live-ownership` with live cadence, reader decisions, paused schedules, and replanning after a concurrent poll change |
| Spectrolite | 21 | `spectrolite-live-vault` and `spectrolite-live-revision-handoff` with multi-hop wiki links, actual renderer preview, and canonical document changes. Compound goals verify PDF evidence across pages before continuing to an authorized vault edit or hold |
| Google Workspace | 61 | Drive inventory and checkpoint review, retained Gmail compose recovery, Calendar resource windows, and compound Gmail-to-Drive goals that confirm approvals and linked assets before an authorized handoff. |

Google's qualification uses the real installed Drive client against a bounded
test transport. News and Spectrolite use the live workspace services for
schedule and editor observations. The managed `open-goal-oracle-CASE`
selectors exercise those outcome checks with a reference execution. The
Mission Control references use the real store in an owned context. Operator
references use the managed terminal driver and an owned hub. Packaged
deployment selectors also need an installed server package configured
through `VIBESTUDIO_MANAGED_DEPLOYMENT_PACKAGE`.

The publishing cases reuse `workspace-publication-preflight` and
`workspace-live-publication-revision`. The harness binds an actual renderer
and website package to an observed workspace revision. It checks readiness
and the exact local publication journal without issuing an external publish.
Eight owned preview goals add bounded provider uploads, interrupted
submission, idempotent retry, hosted-file readback, and approval holds. They
use the installed publisher and durable Host journal with a disposable
provider. The direct provider and journal probe passed. Managed renderer
and RPC verification is tracked in the case evidence audit.
The public-research holdouts use source groups for W3C image accessibility
and Unicode canonical comparison, separate from the training cases about
reserved domains. [W3C decorative images](https://www.w3.org/WAI/tutorials/images/decorative/),
[Unicode normalization](https://www.unicode.org/faq/normalization.html)

To verify a goal against the managed workspace, run its model-free oracle.
Pass `--case` to select goals or `--family` to verify a family from the pinned
catalog. The command groups cases by workspace, baseline, and declared device
drivers. It runs one owned instance at a time and saves each full reference
trajectory. It stops and
recreates the instance every 12 cases by default to bound the lifetime of
renderers, compiler caches, and test state. Completed batches retain their
evidence if a later batch fails. Set `--cases-per-instance` to change the
limit. A value of zero uses one instance per workspace and baseline.
On a continuation, pass `--reuse-references RESULT_DIR` for each previous
result or evidence bundle to skip cases whose existing managed outcome
matches the exact current goal. With `--resume`, the same output directory
can continue from its accepted references after an interruption. Its catalog
must match byte for byte and each attempt retains its own logs. Start a fresh
output directory and omit reuse when retesting a changed fixture or verifier.
With `--resume --drain-file PATH`, creating that file lets the current batch
finish and release its instance before the driver exits. Remove the file
before resuming the remaining batches on that catalog.

```sh
python3 scripts/memory_ledger.py run \
  --unit vibe-open-goal-reference --budget-gb "$REFERENCE_BUDGET_GB" \
  --reserve-gb "$REFERENCE_RESERVE_GB" --hold-budget --no-resumable \
  --workdir "$NATLANG_CHECKOUT" \
  --env "VIBESTUDIO_TEMPLATE_CHECKOUTS=$VIBESTUDIO_TEMPLATE_CHECKOUTS" \
  --env "VIBESTUDIO_MAX_CONCURRENT_BUILDS=1" \
  -- python3 scripts/vibestudio/qualify_open_goals.py \
  --host "$VIBESTUDIO_CHECKOUT" --catalog "$GOAL_CATALOG" \
  --out "$REFERENCE_RESULTS" --case open-goal-reconcile-workshop-update
```

To audit the saved reference results against the pinned catalog, run:

```sh
python scripts/vibestudio/audit_open_goal_references.py "$GOAL_CATALOG" "$REFERENCE_RESULTS" \
  --baseline docs/research/vibestudio-open-goal-baseline.jsonl \
  --coverage "$RELEASE_COVERAGE"
```

Set `RELEASE_COVERAGE` to the `coverage.json` saved in the evidence bundle
for the selected release.

The audit lists missing cases and binds accepted proofs to the current goal
hash and the run's owned context. It records each accepted archive's digest
and the reported System Testing build version. With the baseline, it reports
the remaining corpus shortfall and families without a verified holdout. Add
`--require-complete` to make those gaps or any missing reference fail the
check. For a single family, pass `--family` in place of `--baseline`.
Use `--bundle-out NEW_DIRECTORY` to save the catalog, audit inputs,
and accepted reference traces together. Each copied archive is checked
against the audit's digest, and the saved report uses paths relative to the
bundle. Keep these reference bundles with the corpus verification evidence.
Teacher exports and learner assembly use their own capture directories.

The crosswalk identifies likely skill sources and baseline capabilities by
family. Coverage credit requires an explicit case link and every named check
in that case's accepted reference archive. The report lists the exact cases
that support each capability and skill. It also lists every inventory entry
that still needs evidence.

Pass `--installed-skills "$SKILL_INVENTORY"` to match these links against the
skills observed in each managed workspace. For the full corpus,
`--require-complete` requires both inventories, live observations from all
six workspaces, and at least one verified reference link for each baseline
capability and skill. Each installed skill also needs a reference from its
own workspace.

The first Personal and System live reference runs passed all five planning
cases and exposed missing target dependencies in five native executions.
Native tests now request the declared dependency closure from the host and
record the prepared dependency key in the report. The adapter checks the
requested state, target, suite, and filters, and the harness checks the
workspace state again before issuing a receipt.

The first expansion passed 34 managed reference checks. The result covers all
10 Personal and System suite goals, all 12 News goals, all seven Drive
goals, and all five Spectrolite goals, including the new holdouts. The
Mission Control form suite passed after its manifest declared the DOM
environment and Testing Library peer. The Spectrolite draft oracle
verified the saved edit in the original live renderer and kept the
committed document intact. The MDX exporter now honors the editor's
setting to preserve authored imports without adding generated ones.
All 21 Mission Control and operator additions passed their managed reference
checks. The archive binds the nine board results, four terminal profile
results, and eight deployment results to the pinned goal and owned context.
The deployment run used the installed regular server package at version
0.1.105. The subsequent Examples run passed eleven more document, browser,
and suite cases. All nine recovery references passed against the managed
PubSub and scratch services, including a stale cursor, a concurrent message,
and incomplete replay. All eight initial publishing cases passed against the managed
website renderer, package builder, and local journal. The managed PDF
checks verified the missing-page hold, and two compound references passed
in the original Spectrolite renderer. All 15 Gmail references passed against
native retained compose history and the owned provider transport. The public-research checks
require a working search provider. The managed test reached the default DuckDuckGo provider,
which rejected its automated searches. Configure Tavily, Brave, or Exa in
the managed workspace before running the public-research references.

All 39 Calendar and Gmail-to-Drive cases passed. All nine browser-library and
collection cases passed, including concurrent changes. The first archive
history run passed six cases and exposed a canonical VCS snapshot ordering
failure in 18 fixtures. The fixture now orders paths with the same comparator
as the VCS protocol, and all 18 passed on rerun. Four owned-preview hold
cases passed. Four effectful references exposed loss of typed bytes in a
native filesystem relay. The transport fix passed the direct supervisor
and native worker regression tests, and all four managed preview cases
passed on rerun.
All 15 authoring, 16 System diagnostics, 17 native execution, 26 Personal
automation, and 14 local delivery references passed in managed workspaces.
The audit of the 317-goal catalog accepted 281 references, including 57 held
goals. Its evidence bundle is `vibestudio-open-goal-reference-281-20261120`.
The local delivery verifier checks the sender's observer role and each
recipient's agent role using their exact IDs and handles. Seven visual recovery
cases passed controlled renderer checks and are registered in the same catalog.

The 16 native eval recovery goals, including both held-out goals, passed
managed verification. They cover cancelled and failed evaluations, exact
retry of the original source and inputs, prior-result fallback, and verified
scratch handoff. The 343-goal release adds these goals and ten Personal
browser migration goals, preserving the prior 317 entries. The current
reference evidence covers 307 released goals, including 63 held-out goals.
It also covers 19 goals in staged catalogs. The evidence bundles are
`vibestudio-open-goal-reference-317-20261120` and
`vibestudio-open-goal-google-setup-reference-9-20261120`.
The managed collector now passes the registered goal directly as the Natlang
root's sole argument. Outcome verification also checks the registered goal.

The synthetic assembly check ran with an isolated data registry and the
configured learner tokenizer. It produced one trajectory and 24 supervised
tokens, with both open-goal shares at 100%. A one-reply Luna probe on Tick
passed after sign-in. The first managed attempt exposed invalid service-scope
paths in mounted packages, which were corrected. The model gateway then
waited for Codex credential consent. Resolving the approval in that session
let three actual Luna calls return. The teacher chose the document revision
lambda and hit an unresolved type alias before reading the workspace. The
collector retained the full call tree and verification failure.

The TypeScript project compiler already canonicalized relative source paths.
The virtual loader used by Workerd and the browser now uses the same path
resolution for shared `types.ts` re-exports. The Node filesystem loader
resolves these paths through its filesystem adapter. Llama Server, wllama,
Python native, and SGLang consume the rendered calls from the TypeScript host.
They have no separate implementation of callable-folder source resolution.

The next live run verified the full document recovery on Tick. Luna's
first write saw a stale revision. A dedicated Natlang reconciliation
call compared the original text, proposed change, current text, and fresh
evidence. The final readback confirmed both the requested revision and the
coedit. All five independent checks passed. The capture is registered as
`vibestudio-open-goal-luna-smoke-reconcile-20261120-cqppvh`.
The native trajectory IR records 24 sampled turns across nine invocations,
with five call levels on its longest path. It yielded nine learner chains, 1,735 target tokens,
and 1,712 weighted target tokens with the production LFM3 tokenizer. Both
open-goal shares were 100% for this one-goal smoke. The learner's CPU input
loader packed all nine chains and reproduced the weighted target count.
The longest chain was 25,875 tokens with a 32,000-token packing target. The Tick unit peaked at
7.3 GiB over 3 minutes 10 seconds. No campaign or training was launched.

The [462-goal release](vibestudio-open-goal-pop-pins-462.tsv) preserves all
343 previous rows exactly. It includes 104 held-out goals. Its new lanes cover terminal discovery and
recovery, reconciliation against local and incoming revisions, retained
scratch evidence, Spectrolite knowledge and renderer recovery, native unit
rollout, and Google onboarding. All 14 Google onboarding references passed
in managed Vibestudio. All 13 unit rollout references passed, including the
three cases rerun against the pinned native build. The available evidence
covers 334 distinct goals including 70 held-out goals, with further managed
verification pending.
Google's evidence is registered as
`vibestudio-open-goal-google-setup-reference-14-20261120`.

The staged catalog reached 797 goals with 193 held-out cases. It adds
capability and skill provenance, durable task recovery, client diagnostics,
GitHub readiness, workspace assembly, scheduled automation recovery,
owned HTTP research, compound native eval handoffs, linked vault work,
retained provider approvals, local incident drafts, build provenance,
template migration, local execution width, retained dictation, Adventure
journeys, and specialist handoffs.
The subsequent 898-goal stage adds local visual presentation, deployed actor
and extension interfaces, Personal onboarding and tour handoffs, extension
planning dossiers, and private Tendril route review. The Tendril checks
exercise the real matcher and SQLite-backed Tendril state through a modeled
channel transport. The managed transport still needs verification.
The 970-goal stage also adds retained News evidence, binary asset recovery,
and installed worker package integrity. It has 234 source-separated held
goals. All 842 historical assignment entries still match byte for byte.
The catalog has no ID or source-group overlap with the digest-checked held-v2
and held-v3 snapshots, and no near pairs at the catalog's repetition threshold.
The new News fixtures model article and schedule records. Binary asset
recovery and worker package repair use the production semantic VCS engine in
source replay. Their managed outcomes are tracked independently of source
replay results.
The 998-goal stage adds 28 mobile continuity goals, including six held
goals. It exercises the production startup cache, retained build record
parser, reconnect and trust handling, and mobile lease reconciliation
through local fixture ports. Android hardware and real-server transport
remain separate verification work.
The 1,129-goal source stage adds quiet API readiness, native agent-action
recovery, retained image candidates, durable notifications, and workspace
sharing. It preserves the historical assignments and includes 271 held
goals. The source checks cover typed lambda linking, exact outcome proofs
and source-group separation. The image fixtures exercise retained outputs.
The notification and sharing fixtures exercise durable records and scoped
decisions. Provider generation, desktop toasts, and live account changes
require their own evidence.
The 777-goal checkpoint had distinct texts after normalizing quoted literals
and numbers. All 842 legacy assignment entries remained byte-identical.
The 777-goal checkpoint had zero training ID or source-group overlaps with
the digest-checked held-v2 and held-v3 cases.

Thirteen selected model-free checks passed across Examples, Personal,
Spectrolite, and System. Both previously failing visual recovery checks passed
after the runtime delivered committed state updates to headless panels and
the test granted reload authority for the authorized renderer. The next
managed run added 35 Examples, 22 Personal, 27 Spectrolite, and 57 System
proofs. The combined audit covers 525 distinct goals, including 117 held-out
goals before the assembly and research reruns. The full reference matrix
peaked at 9.5 GiB on Tick. Fifteen assembly reruns and all 12 new held-out
research and editing goals passed in managed Vibestudio. Those research
cases use the native page extractor and retained cache. The following
reference run added 20 independently verified goals. These cover local
incident drafts, provider and dictation continuity, scheduled recovery,
client viewport recovery, and one template migration. The audited 777-goal
release has 572 verified goals, including 142 held-out goals.
Pop reproduced the pinned 317-goal catalog byte for byte from clean detached
checkouts of the host and all eight templates. The source pins and catalog are
under `/home/werg/vibestudio-worktrees/open-goal-release-317.rt37IS` on Pop.
`pins.tsv` records the exact revisions. The catalog SHA-256 is
`b2ed072ab3a9a57632ba1e2ab127ea953a6b2c6b402dd6291aff8cab0eaec5aa`.
The [343-goal source pins](vibestudio-open-goal-pop-pins-343.tsv) reproduce
the release catalog tested on Tick, whose SHA-256 is
`0427ccd200ef1fab0508559eb1564807acaddbef6c1a81f66e57aec685c3528b`.
Pop also reproduced that catalog from the live smoke pins in
`/home/werg/vibestudio-worktrees/open-goal-live-luna-343.WsHCyK`. The
Natlang CLI reported Codex sign-in on Pop. Its host was occupied by a
separate model-serving job at the time of staging.
The 462-goal release also reproduced byte for byte on Pop under
`/home/werg/vibestudio-worktrees/open-goal-release-462.9AiHlQ`. Its catalog
SHA-256 is `5811f79d83aea9db793c1e6825735b98a9ef5447a60fe2434a7b2d8c2464470f`.
That directory includes `train-cases.txt` with 292 selectors backed by managed
proofs and matched to its catalog. Seven of those selectors require an
installed deployment package.
The [updated 462-goal pins](vibestudio-open-goal-pop-pins-462-fixed.tsv) add
headless state delivery, exact capture proof checks, and bounded teacher
instances. Pop reproduced the same catalog and train list under
`/home/werg/vibestudio-worktrees/open-goal-release-462-fixed.XA8cN7`.

The [777-goal release pins](vibestudio-open-goal-pop-pins-777.tsv) passed
the full host commit gate with a 5.8 GiB peak on Tick. Pop reproduced the
catalog and verified the transferred 430-selector training list under
`/home/werg/vibestudio-worktrees/open-goal-release-777.ArOkRz`.
The catalog SHA-256 is
`345666ab124bb2ed1f168920c35b6032a0b5657a0c7348fd7a7feda395d3110c`.
The training selector SHA-256 is
`a3ad8b55ed300d53345cfd76fdd42b678234a5ad88d7d35b5c9bdbd471afcfdc`.
Seven selectors require an installed deployment package. The combined
proof archive is registered as `vibestudio-open-goal-verified-release-777-20261121`.
The follow-up verified all 31 previously failing references, including 16
Adventure goals. The total is now 603 verified goals, including 153 held-out
goals. The 450 training selectors are in `train-cases-450.txt` in the same
Pop release directory. That file has SHA-256
`b6b232cd3b31a35a63b8f8a7b7b85c3f7b1f77bc761ee61d6efdf96a0bf35b85`.
The expanded proof archive is registered as
`vibestudio-open-goal-verified-release-777-extended-20261121`.
The managed follow-up peaked at 8.8 GiB on Tick.

The 970-goal release passed the full host commit gate and was pushed. The
gate peaked at 6.1 GiB on Tick. Pop reproduced catalog SHA-256
`9a50046d00060f9b0825e6c1fdaa77b9cda04104898f5378ae2cbecbb8f5b6df`
and verified the same 450-selector list in the new release directory.
`train-cases-no-device.txt` contains the 440 selectors that require neither
a terminal device nor a deployment package. Its SHA-256 is
`a62a11e916d1754840e5d2716f89ea91cb995947c80f14afa8e975cedbd3d292`.
The managed release batch passed 47 of 48 references and peaked at 9.5 GiB.
The combined audit now has 650 verified goals, including 176 held-out goals.
The proofs are registered as `vibestudio-open-goal-verified-release-970-20261121`.
Use `train-cases-474.txt` for the expanded list on Pop or
`train-cases-no-device-464.txt` to omit terminal-device and deployment
requirements. The files have SHA-256
`4cb74dd485d39ea03df01bdd9747da723bb309207c55fcd8d84f3e104a5c128e`
and `3932ba18d0d4060af83b9db6baa95bd330ad8bfd25d989214a35a3b92687255f`,
respectively. The package-review failure exposed a mismatch with Build V2's
internal dependency validation. The next source release shares that rule
and explains the blocked plan to the caller.

The 1,048-goal release passed the full host gate at 5.5 GiB peak on Tick and
was pushed. Pop reproduced catalog SHA-256
`408061455cbc9bbb40195b382f90306b85bf0edf410450a02c0254872d974354`.
Its `train-cases.txt` contains 474 independently verified training selectors,
and `train-cases-no-device.txt` contains 464. The managed verification run on
Tick reused 638 references for cases that require only the local harness and
started verification of its remaining 393 cases.

The [1,264-goal release](vibestudio-open-goal-pop-pins-1264.tsv) preserves
all 842 historical assignments and previous open goals. It repairs 16
callable paths and pins the Workerd build that resolves imported aliases
in their declaring modules. The full host gate peaked at 5.5 GiB on Tick.
Pop reproduced catalog SHA-256
`aba65637ae07b38f8c541141ad10fc4baa20a256ba98756143e05c1513c0a10f`
and the runtime check passed across 1,079 linked lambdas. The archive
`vibestudio-open-goal-verified-release-1264-20261121` retains 816 managed
reference outcomes, including 204 held goals. Its exact train selector
list has 612 cases. The list for Pop without device or deployment-package
prerequisites has 602 cases.

The [1,320-goal release](vibestudio-open-goal-pop-pins-1320.tsv) adds
26 Personal editorial and 30 retained System release goals. It preserves
the earlier catalog and all 842 historical assignments. The full host gate
peaked at 5.8 GiB on Tick. Pop reproduced catalog SHA-256
`693b0311f666e8694772f43ca9b691174d8190fc85868d41090618bfeb51579a`
and passed the linked runtime check across 1,132 lambdas. The registered
archive `vibestudio-open-goal-verified-release-1320-20261121` carried forward
816 exact managed references and the same 612 train selectors. The initial
model-free sweep contributed its completed batches to the corrected release.

The [composition-corrected 1,320-goal pin](vibestudio-open-goal-pop-pins-composed-1320.tsv)
keeps the same catalog digest. A branch that needs mounted workspace code is
loaded only when all of that branch's required mounts are present. The
release passed all individual template checks with and without System
Testing, plus the combined check. Pop reproduced the corrected bundle and
the System test composition. The registered evidence bundle
`vibestudio-open-goal-composed-release-1320-20261121` has 853 managed reference
outcomes, including 214 held goals. Pop has 639 training selectors, of which
629 have no device or deployment-package prerequisites.

Using the LFM3 learner's tokenizer and request renderer, the corrected
System program's scripted opening measured 21,670 tokens. The previous
recursive scope listing measured 104,742. The new listing gives entry
signatures first. Models can inspect deeper callables and their exact
types with `read_code`. The opening measurement used a scripted callback
and no model weights. Across the seven individual templates, the scripted
opening measured 21,292 to 22,157 tokens.

The [1,479-goal pin](vibestudio-open-goal-pop-pins-composed-1479.tsv) adds
TestKit closeouts, content archive recovery, Phone first-run planning,
first-party Chat continuity, Grimoire scene care, and webhook admission
review. The catalog contains 1,123 train goals and 356 held goals. It keeps
all previous goals and 842 historical assignments. Pop reproduced catalog
SHA-256 `b246d44d03f3e8cdeef71aa694ae82e586f84835a6bb301395f0f5b033b701a1`
and checked both the combined and System test compositions. The registered
archive `vibestudio-open-goal-composed-release-1479-20261121` carries 853
accepted managed reference outcomes and retains the same 639 train
selectors. Of those, 629 have no device or deployment-package prerequisites.

A bounded Luna smoke ran `open-goal-reconcile-workshop-update` in the managed
Examples composition on Tick. Its five independent readback checks passed,
including rejecting a stale write, preserving the concurrent edit and other
documents, and reading back the saved revision. Eleven bound calls imported
as `natlang.teacher_trajectory.native/1` with 28 sampled action records.
CPU-only learner assembly scheduled eight action-bearing chains in three
packs and one step, with 1,596 post-mask weighted target tokens. Both open-goal
shares were 100% for this single-case schedule. The managed smoke peaked at
8.3 GiB and learner assembly peaked at 637 MB. The capture, native IR, and
schedule are under
`/home/werg/data/vibestudio-open-goal-live/luna-composed-smoke-20261121.TjYJ0u`.

The [1,583-goal pin](vibestudio-open-goal-pop-pins-composed-1583.tsv) adds
Git Bridge delivery, Examples adoption, retained development runs and
workspace claim review with the installed Workspace Evidence skill. It
contains 1,201 train goals and 382 held goals. Pop reproduced catalog
SHA-256 `95e5e727c2bfd3f6e381d81ae01dc358b34eeac2644563555be00fafa25620b6`
and checked both the combined and System test compositions. The registered
archive `vibestudio-open-goal-composed-release-1583-20261121` carries 861
accepted model-free managed references. Its 647 training selectors include
637 goals with no device or deployment-package prerequisites. The source
catalog maps all 66 installed skill sources. Each managed reference records
the independent workspace checks for its individual goal.

The [1,707-goal pin](vibestudio-open-goal-pop-pins-composed-1707.tsv) adds
native template conflict triage, retained PTY continuation, owned incident
and release work, workspace scaffolding, and panel launch-state
reconciliation. It contains 1,293 train goals and 414 held goals. Pop
reproduced catalog SHA-256
`82cdfbaf5b8839e955a25889d09af675ff7396408352268e50c5159c3ea80125`
and checked both the combined and System test compositions. The registered
archive `vibestudio-open-goal-composed-release-1707-20261121` carries 914
accepted model-free references. Its 692 training selectors include 682
without device or deployment-package prerequisites.

The [1,812-goal pin](vibestudio-open-goal-pop-pins-composed-1812.tsv) adds
native server-log recovery, route authority review, local RPC settlement,
and CLI gateway and context continuation. It contains 1,372 train goals and
440 held goals. Pop reproduced catalog SHA-256
`5bc2e720024f79b959c488bba17b7188717b0a2dc25aec69eefe9e7b999b5ba9`
and checked the combined and System test compositions. The registered archive
`vibestudio-open-goal-composed-release-1812-20261121` carries 926 accepted
model-free references. Its 701 training selectors include 691 without device
or deployment-package prerequisites. Ten managed canaries passed on Tick
before publication, including the repaired owned-worker and TestKit paths.

The [1,916-goal pin](vibestudio-open-goal-pop-pins-composed-1916.tsv) adds
intent discovery from workspace evidence, same-turn Eval recovery, durable
GAD message and participant review, and live Tendril CDP and Build V2 work.
The catalog contains 1,450 train goals and 466 held goals. Pop reproduced
SHA-256 `b19d4efad6b8899fe4b8ec3b8d95b4d16fe0fd598f364918c9fe7e15fd0e7ee4`
at `/home/werg/vibestudio-worktrees/open-goal-composed-1916.vUphMG` and
verified the combined and System managed compositions. The registered
`vibestudio-open-goal-composed-release-1916-20261121` archive contains
989 accepted references, including 747 train selectors. The local selector
file contains 737 of those. Full Host infrastructure verification on Pop
waits for the ledger to admit its memory budget. The same staging procedure
verified those native Host dependencies on Tick on the 1,812-goal release.

The [2,224-goal pin](vibestudio-open-goal-pop-pins-composed-2224.tsv) adds
native workspace configuration, Android delivery, resident messaging recovery,
board migration, coordinated document changes, and a broader set of writing
and presentation goals. The catalog contains 1,674 train goals and 550 held
goals across 107 families. Its SHA-256 is
`66291e428c969effff062d8c730031e46744c7d9745c76d12137520c2165fa81`.
The registered `vibestudio-open-goal-composed-release-2224-20261121` archive
contains 1,077 accepted model-free references and directly exercises 141 of
the 151 baseline agentic capabilities. It includes 788 training selectors,
778 of which require only a local workspace. The archive retains the prior
release's accepted training selectors.

Pop verified the catalog, selector hashes, and native Host build in
`/home/werg/vibestudio-worktrees/open-goal-composed-2224.LymeVV`. The release
directory contains the `native-host-verified` marker and both selector files.

This pin enables `qualify_open_goals.py --retain-trajectories`. The managed run
keeps each selected test's full evidence as it finishes the batch. Two native
notification cases verified that path on Tick. The archived Luna document
trajectory also rebuilt against this catalog, yielding eight action chains,
three packs, and 1,596 weighted target tokens in a CPU-only learner schedule.
A ground-truth oracle receipt verifies a workspace outcome. A teacher
trajectory additionally records the sampled model decisions that the learner
can supervise.

The [2,269-goal pin](vibestudio-open-goal-pop-pins-composed-2269.tsv) adds
45 open goals for recorded change rationale, counteractions, workspace unit
design, and visual identity. Its SHA-256 is
`bd478f07d0105917951c9bd7faaeee7d13f4b8bb9ceed0f1afa1069ca3cc7887`.
It contains 1,707 train and 562 held goals. All 842 historical assignments
remain byte-identical. The registry entry
`vibestudio-open-goal-composed-release-2269-20261121` contains 1,108 accepted
references, including 811 train goals and 801 local train goals. The pinned
archive directly covers 144 of the 151 baseline agentic capabilities.

Seven managed checks passed for the new provenance and unit paths and the
Grimoire Portfolio repair. The active local sweep uses complete receipts from
prior batches to resume without repeating accepted cases. The collector now
provisions each declared device profile as part of its workspace group, so
readiness checks, execution, and evidence export use the same attachments.

Release staging also builds the pinned Natlang Node modules used for the
independent semantic reviewer. Pop verified the reviewer imports for the
2,224-goal release and reported a configured Codex credential without a model
turn. Cases that require a review still need the explicit choice of a reviewer
model.

The 2,294-goal release adds 20 native image lifecycle goals and five passive
Android runtime audit goals. It contains 1,724 train and 570 held goals, with
catalog SHA-256 `3a124b9e569bb630c4f678cedeb2fe6e4c0c1065c2d8d06abbfebe392c8dad03`.
The registered `vibestudio-open-goal-composed-release-2294-20261121` archive
contains 1,197 accepted references and 881 training selectors, including 871
that require only a local managed workspace. It directly covers 145 of 151
baseline agentic capabilities. [Commit pins](vibestudio-open-goal-pop-pins-composed-2294.tsv)

Five managed image cases verified the production async job, retry and save
path, with deterministic image bytes supplied at the provider boundary.
Actual external generation and image quality still need their own check.
Passive Android references cover connection, a missing package, a stopped
app, a stale target, and a running process with a device screenshot. Their
source fixtures passed. The managed variants require an assigned disposable
emulator in the app state specified by the case.

Public search receipts now bind the selected provider and result URLs to the
citation readback. Owned HTTP fixtures verify the native public-session
fetch, bounded cache continuation and citation steps. Live dynamic public
search requires a working search provider in the managed workspace.

A single managed System instance completed 24 model-free references under
the 9.5 GiB cap. The combined Personal and System test unit peaked at
8.6 GiB. Routine sweeps keep the existing 12-case recycle interval.

The 2,320-goal release adds owned automation activation, Calendar runbook
reconciliation, and News editorial review. The catalog has 1,742 train
and 578 held goals and preserves all 842 historical assignments. Its SHA-256
is `5b8b27a990541fd5fa3e689953d16c47790b0f7d44a8afca2876bf77d358bfc9`.
The registered `vibestudio-open-goal-composed-release-2320-20261121` archive
contains 1,248 accepted references. It has 917 train selectors, including
907 for local workspaces. Managed references exercise all 66 skills somewhere
in the corpus and 145 of the 151 baseline agentic capabilities. The complete
audit also tracks each skill in each workspace and the 2,164-reference floor.
The continuing sweep prioritizes families that still need a local held
reference. [Commit pins](vibestudio-open-goal-pop-pins-composed-2320.tsv)

Pop verified the exact catalog, case selector hashes, native Host dependencies,
and pinned Natlang reviewer in
`/home/werg/vibestudio-worktrees/open-goal-composed-2320.fTYaju`. It verified
all 14 infrastructure package input digests against a prepared release and
revalidated the copied output hashes through the Host's own build checker.
Staging fit while the held evaluation continued. The Pop credential status
reported a configured Codex login.

The archived Luna capture rebuilt against this exact catalog. The learner's
CPU-only schedule contains eight action chains and 1,596 weighted target
tokens. The gate still requires at least 80% open distinct goals and 80% open
weighted target tokens for the actual train schedule. [Training pipeline](#bounded-multi-case-capture)

The 2,324-goal release adds four assigned public briefs, including a held
review that withholds a website-wide conformance claim. All four passed
against live pages in the managed Examples workspace. This path uses only
the assigned URLs and verifies actual text, cache, citations, and the
resulting decision. It requires public HTTPS access. Dynamic public search
still has its own provider prerequisite.

The catalog contains 1,745 train and 579 held goals. Its SHA-256 is
`de3601ae32d387aceb46c2989b38606387d2cea2c8a642103b3bc44b35d16bd4`.
It preserves every released 2,320-goal row and all 842 historical
assignments. All 15 installed compositions compiled. The combined
composition links 1,809 Natlang lambdas and 236 explicit qualified
references. [Commit pins](vibestudio-open-goal-pop-pins-composed-2324.tsv)

The registered `vibestudio-open-goal-composed-release-2324-20261121`
archive contains 1,404 accepted reference outcomes, 1,028 train selectors,
and 1,015 train selectors for local managed workspaces. Its registry
manifest SHA-256 is
`c61d19322813c64a85f5710c262d8831b39d1d6588b7d5f4e64b0e84248a3763`.
These references exercise all 66 skill sources somewhere in the corpus
and directly cover 146 of the 151 baseline agentic capabilities. Each
family with a local managed reference path has an accepted held outcome.
The broader local sweep keeps adding evidence for individual cases.

The remaining five baseline capability checks need live Android access.
Use an assigned disposable emulator. Installation tests expect a clean
start with the internal app absent. Pairing needs the desktop attachment
and a compatible installed app. The passive runtime audits need the
device state named by each goal, including absent, stopped, and running
app states. These tests admit evidence against the exact assigned
emulator and state.

The archived Luna trajectory rebuilt against this release into eight
action chains and 1,596 weighted target tokens. Both open-goal shares for
its CPU-only schedule were 100%. Of that weight, 1,446 tokens came from
nested invocations at depths 1 through 4. The learner checks both shares
on the complete schedule of a training run.

Pop verified the catalog, selector hashes, all 14 Host infrastructure
packages, native Host runtime dependencies, and the pinned Natlang reviewer
in `/home/werg/vibestudio-worktrees/open-goal-composed-2324.GGDHc7`.
The verified local selector file is `train-cases-no-device.txt`. Two of
its goals require a separate semantic-review model. The managed collector
preflight is queued behind Pop's active held evaluation.

The 08:00 local sweep checkpoint contains 1,644 accepted outcomes.
It is registered as `vibestudio-open-goal-2324-sweep-20261121t0800` with
manifest SHA-256
`3189a84763496ba4ad6d50f4a2cdf9c339c314024cada359cf72f4621b7d8378`.
In the same Pop release, `selector-snapshots/managed-1644-20261121T0800/`
contains a verified superset of the initial selectors, with 1,215 training
goals and 1,202 local training goals. Pop's pinned materializer also checked
the transferred Luna manifest and emitted 25 eligible native training
decisions.

The wider System sweep found two log-follower cases where Workerd emitted
several lines in a single pipe read. The Host had indexed each read as one
record. The corrected Host frames complete UTF-8 lines before assigning
server-log records, with a bounded buffer for long lines. Both failing
cases and two complete-window regressions passed against the corrected Host.
The Workerd and server-log suites passed 106 tests.

Pop verified the repaired Host, native dependencies, reviewer and
unchanged catalog in
`/home/werg/vibestudio-worktrees/open-goal-composed-2324-framing.Q877LO`.
The registered `vibestudio-open-goal-2324-log-framing-20261121` archive
contains 1,740 accepted outcomes, 1,284 training selectors, and 1,271
local training selectors. It preserves the 08:00 selector snapshot in
full. The managed collector preflight for this pin is queued behind the
same Pop held evaluation. [Corrected commit pins](vibestudio-open-goal-pop-pins-composed-2324-framing.tsv)


## Semantic application entries (2026-11-23)

The full 2,327-task catalog now binds typed application functions, not a universal
`runUserGoal` router or `UserGoalReport`. All original goal hashes, task identities,
source groups and splits are preserved (1,748 train; 579 test). The shared library
declares 116 entry kinds across 27 program roots; this catalog uses 114 entry
programs. Existing specialist lambdas, callable-folder scopes and cross-domain
composites remain available. An entry descriptor binds the actual named inputs,
exact JavaScript JSON bytes, callable and application result type.

Outcome checks verify the requested effects and required returned values against
workspace evidence. They no longer require incidental tool order, exact read
counts or generic final reports. Scope protection, claimed source evidence and
application-required answers remain checked. Additional authorized exploration,
retries and concurrent identical reads are permitted. Known incorrect optional
root returns remain in the raw trace but are excluded from supervised targets;
a completion mixing a valid action and an incorrect return is retained without
inventing a safe token boundary.

Historical captures retain their original program identity. Reassessment adds
immutable evidence; it never rewrites the original outcome or labels partial
work as task success. The first recovered cohort contains ten captures and 42
independently checked decisions. Its additive receipt is
`recovered-independent-decisions-20261122/receipt-v2.json` in the DGX campaign
archive. Native trajectory IR remains the training representation. The existing
live ingester and student reasoning fill consume admitted IR; raw teacher
reasoning is not a student target.

Compiler checks passed all 116 entry signatures. Focused validation passed 120
Host tests, 32 system-testing tests, 30 shared-library/provider tests and 61
capture-loader tests (122 across loader, ingestion and SFT after integration fixes). Release assembly and live cutover are separate operational
steps; these source checks do not claim all catalog tasks have succeeded live.


The full Luna campaign keeps archival `--wave-size` separate from runtime
lifetime: `--cases-per-instance` defaults to 12 even for a 64-case wave.
`--deployment-concurrency` defaults to 2 for waves containing installed
deployment targets; ordinary waves retain the requested `--concurrency`.
Override these limits explicitly when the machine budget permits.
