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
export VIBESTUDIO_TEMPLATE_CHECKOUTS
cd "$VIBESTUDIO_CHECKOUT"
pnpm --silent natlang:teacher --list > "$GOAL_CATALOG"
python "$NATLANG_CHECKOUT/scripts/memory_ledger.py" run \
  --unit vibe-open-goal-smoke --budget-gb "$SMOKE_BUDGET_GB" \
  --reserve-gb "$SMOKE_RESERVE_GB" --no-resumable --workdir "$VIBESTUDIO_CHECKOUT" \
  --env "VIBESTUDIO_TEMPLATE_CHECKOUTS=$VIBESTUDIO_TEMPLATE_CHECKOUTS" \
  --env "VIBESTUDIO_MAX_CONCURRENT_BUILDS=1" \
  -- pnpm natlang:teacher --case open-goal-reconcile-workshop-update \
  --out "$CAPTURE_DIRECTORY"
```

Wait for the unit to exit successfully and keep the entire capture directory.
For assembly on a different machine, copy that directory and the exact JSONL
catalog from Pop. The manifest's bound validation, call, and readback artifacts
use paths relative to the manifest. Use a machine with the Neuralese Python
environment and point `CAPTURE_DIRECTORY` and `GOAL_CATALOG` at the transferred
copies. Activate that Python environment before running the commands below.
The learner needs the tokenizer and chat template from the starting model.
Only tokenizer assets are loaded for assembly. Use that machine's held-v3
manifest for `HELD_MANIFEST`, with the referenced held-v2 exclusion available.
Place the capture, work, and learner output directories under a configured
logical data root. Use fresh directories and set a unique `SMOKE_DATASET_ID`
for this capture.

```sh
cd "$NATLANG_CHECKOUT"
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

The catalog declares 279 open goals across six workspace templates, with 218
training goals and 61 source-separated holdouts. The same Base entry point
receives the goal across these workspaces. It links the template's installed
Natlang subtree and scopes the template services to the functions that use
them. The workspace and the reference checks are selected by the harness.

| Workspace | Goals | Baseline capability and open-goal coverage |
| --- | ---: | --- |
| Examples | 70 | Public evidence and document recovery, ordered CDP changes and readback, declared suite verification, publishing, native PDF evidence, and Natlang program discovery, compilation and stale source recovery |
| Personal | 74 | Declared tests, Mission Control, browser-library-to-collection work, archived decisions, and coordination across authorized monitors. Exact readback checks recovered references, historical authority, and interrupted scheduling changes. |
| System | 41 | Declared tests, hub and terminal operations, retained conversations and cursor recovery, and Host diagnostic windows across boots, failures and owned incident handoffs |
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
catalog. The command groups cases by workspace and baseline, runs one owned
instance at a time, saves each full reference trajectory, and stops the
instance before moving to the next group.

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
  --coverage docs/research/vibestudio-open-goal-coverage.json
```

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
All 15 authoring and 16 System diagnostics references passed in managed
workspaces. The preserved evidence bundle
`vibestudio-open-goal-reference-223-20261120` includes the 279-goal catalog,
223 accepted references, and 47 verified holdouts.

The synthetic assembly check ran with an isolated data registry and the
configured learner tokenizer. It produced one trajectory and 24 supervised
tokens, with both open-goal shares at 100%. Bounded Luna provider probes on
Tick and Pop returned expired OAuth. The live teacher capture through
learner assembly remains unverified.
Pop reproduced the prior 72-case catalog byte for byte from the pushed host
and System Testing revisions.
