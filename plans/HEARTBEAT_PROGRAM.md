# The hourly check-in as a natlang program (review item P1)

Status: implemented in `applications/heartbeat/` (2026-10-09, shipped in advisory mode; implemented before the owner's
review, which is now an after-the-fact review in plans/OWNER_REVIEW.md). Where the build differs from the draft below:

- The "Is<T, ...>" checks of section 10 that need the run's own readings, the allowlist or the machine's resource numbers
  (quote occurs in the reading, proposal valid against the allowlist, `idle_resources` against thresholds) are exact
  host verifiers (`verify.ts`, `actions.ts`) whose problems go back to the function through a `feedback` argument;
  the one-sentence and at-most-three-sentences predicates are checked the same way, not by the judge.
- `actions.json` also holds the settings (stage, modes, thresholds, repairs); the stage lives there so only the owner changes it.
- The inbox is read with `coord.py inbox --json` (new, read-only, no cursor) rather than by importing `Store`; question 6 is
  answered by using `scripts/coord.py`.
- Each part is `pluggable` (`crisp|nl|shadow`); the crisp side is `ladder.ts` and is also the degraded mode.

Source: plans/NATLANG_NATIVE_REVIEW.md, row P1. The hourly check-in is done today by agents in Claude sessions. It
decides run health, cause, resume, replace or wait, and the next step. That judgment leaves no trace that the
specializer or training can use, and it depends on a session staying alive.

## 1. What happens today

- **Rule.** `AGENTS.md` ("Machine ownership and coordination"): run the coordination inbox check at session start,
  before changing shared resources and at every monitoring cycle; check at least once per 50-minute work or sleep
  cycle. The memory note "Autonomous heartbeat" (owner, 2026-10-05) adds: once an hour, diagnose and fix, never leave
  DGX resources idle, queue the next most valuable step.
- **Scheduling.** A Claude session creates an hourly `CronCreate` prompt at an off-minute. The cron belongs to that
  session; it ends with the session. The project hooks (`.claude/settings.json`) only run
  `scripts/coord.py brief` at session start and on each prompt, which prints a one-line notice when something is
  unread. Nothing runs when no session is alive.
- **Evidence the agent reads by hand.** `python3 scripts/memory_ledger.py status` (claims, headroom, hold flags),
  `systemctl --user` units (`natlang-*`), unit logs (`/tmp/*.log`, `~/.local/state/natlang/*.log`), GPU use
  (`gpu_usage()` in `scripts/memory_ledger.py:139`), teacher load (`:8082` metrics, the window file
  `runs/research-teacher-20261006/teacher-window.json`), gate reports (for example the
  `natlang.neuralese-foundation-control/1` JSON written by `training/neuralese/natlang_neuralese/eval/foundation.py:125-128`,
  field `token_aligned_reference_passed`), and the coordination mailbox (`scripts/coord.py`).
- **Decisions the agent makes.** Is each run healthy, what is the cause when it is not, resume or replace or wait, and
  what is the next step. Process control (pause, stop, replace) is the owning session's call (memory:
  "Process control is mine"; `AGENTS.md`: stop by unit name or explicit PID, never by pattern).
- **`AGENTS.md` names `scripts/coordination_inbox.py check --ack`; this checkout has `scripts/coord.py inbox --ack`
  only.** The collector below uses `coord.py` and notes the rename for the owner (question 6).

## 2. Design in one paragraph

A natlang program with three typed functions, `triageInbox`, `diagnoseRun` and `planNext`, reads crisp read-only
evidence and returns a `HeartbeatReport`. A crisp allowlist (data) defines the actions the report may propose, with
their reversibility and owner. In advisory mode nothing is applied. After a measured period, only actions that the
allowlist marks reversible may be applied automatically; process control is only ever proposed to the owning
session. A systemd user timer runs it on each machine, independent of any Claude session. Every call is recorded in
the machine call store, so the specializer compiles the routine cases and the trajectories become training data.

## 3. Policy and mechanism

| Part | Decision | Why |
| --- | --- | --- |
| Reading units, logs, ledger, gates, mailbox | crisp collectors, read-only | The outside world; exact. |
| Minutes since progress, error candidates, gate field extraction | crisp | Arithmetic and pattern extraction over readings, so the model reads numbers and short candidate lists. |
| Is the run healthy, and why not | fn `diagnoseRun` | A judgment over evidence. |
| What the mailbox asks of this machine | fn `triageInbox` | A judgment over messages. |
| What to do now and next | fn `planNext` | A judgment over diagnoses, obligations and resources. |
| Which actions exist, are they reversible, who owns them | data (`actions.json`) | Policy the owner edits; the model never declares reversibility. |
| Proposal validity (action in allowlist, params match schema, target exists) | crisp | Exact verifier of the stage output. |
| Quote and id citations exist in the readings | crisp | Exact verifier. |
| Apply an auto-allowed action | crisp, argv from the allowlist | Exact; receipts record before and after. |
| Compare with the agent's decisions | crisp | Exact set comparison; see section 7. |
| Schedule, lock, overlap | host (systemd timer, flock) | Mechanism. |

State model: a heartbeat is decide-on-a-snapshot, then a pure bounded apply. The snapshot is the `Evidence` value
(all readings with their hashes). The decisions are functions of it. The effects are data (`Proposal[]`). Applying is
a separate step that only touches allowlisted actions. Derived values form a DAG: readings, run evidence, diagnoses,
obligations, plan, applied receipts, comparison. No stage loops until a condition holds; repetition is per run and
per message.

## 4. Crisp evidence collectors

All collectors are read-only, run with a timeout, and return a `Reading`. A failing collector returns
`ok: false` with the error text; the functions see the failure as evidence (a missing reading is a finding).

```
type Reading = {
  id: string,                 // "<source>:<subject>:<hhmm>"
  source: "ledger" | "unit" | "log" | "gpu" | "gate" | "inbox" | "git" | "teacher" | "disk" | "peer-status",
  at: string,                 // ISO time of collection
  ok: boolean,
  text: Untrusted<string>,    // output or the parsed fields, bounded
  truncated: boolean,
  sha256: string,
}
```

| Collector | Command or file | Output used |
| --- | --- | --- |
| `ledger` | `python3 scripts/memory_ledger.py status` (JSON: `available_gb`, `headroom_gb`, `claims{class, budget_gb, used_gb, peak_gb, outstanding_gb, hold_budget, command}`) | claims, headroom |
| `unit` | `systemctl --user show <unit> -p ActiveState -p SubState -p Result -p ExecMainStatus -p ActiveEnterTimestamp` for each watched unit, plus `systemctl --user list-units 'natlang-*' --all` | state per unit, failed units not on the watch list |
| `log` | the last 80 lines and the mtime of each watched log | `log_tail`, `minutes_since_progress` (time since the last line that matches the entry's `progress_marker`), `error_candidates` (the last five lines matching `Error|Traceback|OOM|refused|Killed|CUDA`) |
| `gpu` | `gpu_usage()` of `scripts/memory_ledger.py` (DGX); `nvidia-smi --query-gpu=...` (Pop) | utilization, memory |
| `gate` | each watched gate JSON, parsed; only the keys named in the watch entry's `gate_fields` | pass or fail fields, schema name |
| `inbox` | `scripts/coord.py` `Store.messages()` filtered by `addressed` and `resolutions` (read-only; no cursor change) | unread and open messages for this machine |
| `peer-status` | `coord.py status` (the other machine's page) | what the peer reports running |
| `teacher` | `:8082/metrics` (`vllm:num_requests_running`, `vllm:num_requests_waiting`), the teacher window file | load, window |
| `git` | `git status --porcelain=v1 -b`, `git log --oneline -5`, `git rev-list --count origin/main..HEAD` (no fetch) | local divergence |
| `disk` | `df -h` of the data roots, `/proc/meminfo` `MemAvailable` | space, memory |

Watched runs are declared in a data file, not guessed. The owning session adds an entry when it starts a long job
(the same moment it registers the job with the ledger):

```
type WatchEntry = {
  run_id: string, machine: "dgx" | "pop", owner: string,       // owning session or agent name
  unit: string,                                                 // systemd unit name
  kind: "training" | "generation" | "eval" | "build" | "service",
  expected: "finishes" | "serves",
  log: string | null, progress_marker: string,                  // a regular expression for a progress line
  stall_after_minutes: number,
  gate_paths: string[], gate_fields: string[],
  next_steps_doc: string | null,                                // where "what comes after this run" is written
}
```

A run on the machine that is in the ledger or `systemctl` but has no `WatchEntry` appears as an `unwatched`
finding in the report so the owner sees it.

## 5. The natural-language functions

All three read `Untrusted<string>` for log, gate and message text. Each is a `.nl` file with the contract shown.

### `diagnoseRun`

```
args: run: RunEvidence
returns: RunDiagnosis

type RunEvidence = {
  entry: WatchEntry,
  unit: { active_state, sub_state, result, exit_status, started_at },
  minutes_since_progress: number | null,
  log_tail: Untrusted<string>, error_candidates: Untrusted<string>[],
  gates: { path, schema, fields: Record<string, unknown> }[],
  ledger_claim: { budget_gb, used_gb, peak_gb, hold_budget } | null,
  headroom_gb: number, readings: { id, source, ok }[],
}
type Health = "progressing" | "stalled" | "finished-ok" | "finished-failed" | "blocked" | "gate-failed" | "unknown"
type Cause  = "none" | "admission-refused" | "out-of-memory" | "input-missing" | "code-error" | "gate-failed"
            | "resource-contention" | "stalled-no-error" | "external" | "unknown"
type RunDiagnosis = {
  run_id: string, health: Health, cause: Cause,
  evidence: { reading_id: string, quote: Is<string, "occurs in the reading with reading_id"> }[],
  summary: Is<string, "one sentence naming the run, its health and the cause">,
  confidence: "high" | "medium" | "low",
}
```

Steps:

1. Read `unit.active_state` and `unit.result`. When the unit is `failed` or exited with a non-zero `exit_status`,
   health is a candidate `finished-failed`. When it exited with 0 and `entry.expected` is `finishes`, health is a
   candidate `finished-ok`.
2. When the unit is `active` and `entry.expected` is `finishes`: compare `minutes_since_progress` with
   `entry.stall_after_minutes`. At or above it, health is a candidate `stalled`; below it, `progressing`.
3. Read every gate in `gates`. When a field named by the schema reports failure (for example
   `token_aligned_reference_passed` false), health is `gate-failed` and cause is `gate-failed`; quote the field.
4. For `finished-failed`, `stalled` and `blocked`, read `error_candidates` from the last to the first and pick the
   line that explains the stop. Match it to a cause: a refusal line to `admission-refused`; "Killed" or an OOM line
   to `out-of-memory`; a missing-file line to `input-missing`; a traceback to `code-error`; high `used_gb` with low
   `headroom_gb` to `resource-contention`. When no line explains it, use `stalled-no-error` (for a stall) or
   `unknown`.
5. Write `evidence`: one to three entries, each a reading id and a quote copied from that reading.
6. Write `summary` in one sentence and set `confidence` high when a quoted line names the cause, medium when the
   cause follows from numbers, low otherwise.

### `triageInbox`

```
args: messages: CoordMessage[], machine: "dgx" | "pop", watched: { run_id, unit }[]
returns: InboxTriage

type CoordMessage = { id, from, to: string[], kind: "note" | "request" | "decision" | "reply" | "close",
                      subject: string, body: Untrusted<string>, reply_to: string | null, urgent: boolean, sent_at: string }
type Obligation = {
  message_id: string,
  kind: "answer-request" | "adopt-decision" | "acknowledge" | "inform",
  what: Is<string, "one sentence stating what the message asks of this machine">,
  affects: string[],                    // run ids or paths named in the message
  reply_needed: boolean,
}
type InboxTriage = { obligations: Obligation[], digest: Is<string, "at most three sentences"> }
```

Steps:

1. For each message, read `kind`, `subject` and `body`.
2. A `request` addressed to this machine is `answer-request`; say what is asked in one sentence.
3. A `decision` is `adopt-decision` when its body changes how this machine runs or builds something; otherwise
   `acknowledge`.
4. A `note` is `inform`. `reply` and `close` messages are `inform` unless they answer a request this machine sent.
5. List in `affects` the run ids from `watched` and any path or commit the message names.
6. Set `reply_needed` true for `answer-request` and for urgent messages.
7. Write `digest`: the obligations in priority order, at most three sentences.

Acknowledgement stays with the session. The program never advances a reader cursor (`coord.py inbox --ack`), since
"acknowledgement is not adoption" (`plans/MACHINE_COORDINATION.md`).

### `planNext`

```
args: diagnoses: RunDiagnosis[], triage: InboxTriage,
      resources: { headroom_gb, gpu_utilization, teacher_load, unwatched: string[] },
      next_steps: Untrusted<string>[],        // the text of each watched run's next_steps_doc
      allowlist: ActionSummary[],             // id, what it does, params schema, applies_to causes
      history: { cycle: string, proposals: Proposal[], agent_actions: ActionRef[] }[]   // last cycles, for repetition
returns: Plan

type Proposal = {
  action: Is<string, "an id in allowlist">,
  target: string,                         // a run id, unit name, message id or path
  params: Record<string, string>,
  why: Is<string, "one sentence citing a diagnosis or obligation">,
  cites: string[],                        // run ids and message ids
  expected_effect: string,
}
type Plan = { proposals: Proposal[], idle_resources: Is<string, "empty when no resource is idle, otherwise names the idle resource">,
              summary: Is<string, "at most three sentences for the status page"> }
```

Steps:

1. Order the diagnoses: `finished-failed`, `blocked`, `gate-failed`, `stalled`, then the rest.
2. For each diagnosis that is not healthy, choose from `allowlist` the actions whose `applies_to` includes its cause,
   and write one proposal each: target the run, params from the schema, `why` citing the diagnosis. The first choice
   is the action that gathers more evidence; the next is the action that changes the run (a relaunch through the
   ledger, a stop).
3. For each obligation with `reply_needed`, propose the reply action targeted at its message.
4. Check the resources. When `headroom_gb` is large, `gpu_utilization` is low and `teacher_load` is low, set
   `idle_resources` to the name of the idle resource and propose the first entry of the relevant `next_steps`
   through the allowlist's queue action. Otherwise leave `idle_resources` empty.
5. Drop a proposal equal to one that `history` shows was proposed in the last cycle and followed by no change; replace
   it with the next action for the same cause.
6. Write `summary` for the status page: what runs, what failed, what is proposed.

The proposal carries no reversibility, owner or auto flag. The crisp applier fills them from `actions.json`.

## 6. The action allowlist

`actions.json` (data, owned by the owner; the program may read it, never edit it). Every proposal's `action` is an
id in this table. Reversibility and ownership are properties of the action, set here.

| Action id | Command (argv fixed in the file) | Reversible | Owner | Auto-apply |
| --- | --- | --- | --- | --- |
| `record-heartbeat` | append the report to `runs/heartbeat/<date>.jsonl` | yes (append-only record) | this program | yes, always |
| `set-status-page` | `coord.py status --set` with the plan summary | yes (overwritten next time) | this machine's session | yes, after promotion |
| `release-cache` | `memory_ledger.py release-cache` | yes (cache refills) | this machine's session | yes, after promotion |
| `schedule-recheck` | one-off `systemd-run --user --on-active=<m>m` of this program | yes (the transient timer can be stopped) | this program | yes, after promotion; minutes below the timer interval |
| `read-more` | an allowlisted read-only command, for example `journalctl --user -u <unit> -n 200` | yes (reads) | this program | yes, after promotion |
| `reply-note` | `coord.py reply <id> -m <text>` | no (delivered) | the session | proposal only |
| `send-note` | `coord.py send --kind note` | no (delivered) | the session | proposal only |
| `ack-inbox` | `coord.py inbox --ack` | no (moves a reader cursor) | the session | proposal only |
| `relaunch-run` | `memory_ledger.py run --unit ... --budget-gb ... -- <declared argv>` | no (process control) | the owning session | proposal only |
| `stop-unit` | `systemctl --user stop <unit>` | no (process control) | the owning session | proposal only |
| `adopt-unit` | `memory_ledger.py adopt ...` | no (process control) | the owning session | proposal only |
| `queue-next` | start the first entry of a run's next-steps document through the ledger | no (process control) | the owning session | proposal only |
| `edit-code` | (none: the program emits a diagnosis and cites files; a session edits and commits) | n/a | the session | never proposed as a command |

Rules the applier enforces in code, regardless of what a plan says:

1. Only actions whose row says "yes" in the auto column and whose promotion stage has been reached are applied.
2. Process control (`relaunch-run`, `stop-unit`, `adopt-unit`, `queue-next`) is always a proposal to the owning
   session. The program writes the proposal into the report and, when the owner asks for it, into a coordination
   request addressed to that session; it never executes it.
3. Parameters are validated against the row's schema (unit names must be on the watch list; minutes must be below the
   timer interval; texts are bounded).
4. An applied action writes a receipt (before state, argv, exit code, after state) to the heartbeat record.
5. Each machine's program applies only to that machine. The other machine's page and mailbox are read-only inputs.

## 7. Advisory mode first, and the comparison with agent decisions

Stages (owner decides when to advance; the numbers are proposals):

0. **Shadow.** The program runs hourly, applies only `record-heartbeat`, and writes `runs/heartbeat/latest.md`. The
   agent check-ins continue unchanged and do not read the report before deciding.
1. **Advisory.** Agents read the report at the start of a check-in and record whether they followed it. Still only
   `record-heartbeat`.
2. **Reversible auto-apply.** The rows marked "yes, after promotion" apply automatically. Process control stays a
   proposal.

Comparison (crisp, run after each cycle's agent check-in):

- **Agent actions** are derived from observable effects and mapped to allowlist ids: ledger events since the last
  cycle (`state['events']`) become `relaunch-run`, `stop-unit`, `adopt-unit` and `release-cache`; commits and pushes
  by the machine's sessions are `edit-code`; `coord.py log --from` gives `send-note` and `reply-note`; changes to the
  status page give `set-status-page`. Sessions may also write an explicit line (`heartbeat.py record --action ...`)
  when they act without a visible effect.
- **Match** a proposal to an agent action when the action id and target are equal in the same cycle or the next one.
- **Metrics per cycle and per cause:** proposals the agent also took (agreement), proposals the agent did not take
  (extra), agent actions the program did not propose (missed), and runs where the program said `progressing` or
  `finished-ok` while the agent acted on them (a disagreement flagged for review).
- **Outcome label** (crisp, from the next cycles): after a proposed or taken action, did the run's health improve
  (`stalled` to `progressing`, `finished-failed` to `progressing`) within the next three cycles?
- **Promotion criteria** (proposed, owner's call): stage 1 after at least three consecutive days of cycles in which
  every `finished-failed` and `stalled` run received a diagnosis whose cause matches the agent's; stage 2 for a given
  action when its proposals have matched an agent action in a stated share of cycles and never preceded a degraded
  outcome. The owner sets the share and the minimum number of cycles.

## 8. Scheduling

- **Timer.** A systemd user timer, `natlang-heartbeat.timer` per machine, `OnCalendar=*-*-* *:17:00`,
  `Persistent=true`, activating `natlang-heartbeat.service` (a oneshot). The minute is off the hour to avoid the
  cache-hygiene timer and other jobs (`natlang-development-cache-hygiene.timer` already runs every ten minutes).
  The service takes `flock` on `.coordination/heartbeat.lock` so cycles do not overlap, and registers a small
  ledger claim (`--budget-gb 0.5 --reserve-gb 0`), since "everything that loads a model goes through the ledger"
  even though the heartbeat itself is a client of an already running executor.
- **Executor.** The program calls the executor that the specializer uses (`:8083`, the small student/executor
  endpoint), not the saturated teacher (`:8082`). Before calling, it applies the idle check the specializer uses
  (`waitForIdle`, `applications/specializer/main.ts:101-123`). When no executor answers, the program degrades: the
  collectors run, a crisp status ladder (unit state, progress age, gate fields) produces health, and the report
  says that natural language was unavailable.
- **Agent cron.** The Claude `CronCreate` hourly prompt stays for agents during stages 0-1, changed to "read
  `runs/heartbeat/latest.md`, act, record your actions". `CronCreate` is session-bound and cannot read another
  machine's units; a cloud routine (`schedule` skill) cannot read local units either. The machine-native timer
  is the right host for the program.
- **Ownership.** Each machine's agent owns its timer (`AGENTS.md`: the Pop agent owns Pop execution, the DGX agent
  owns DGX execution). Starting, stopping or editing the unit is process control: by unit name.
- **Cadence.** The timer is hourly; `schedule-recheck` can request a one-off earlier run after a relaunch.

## 9. Traces feed the specializer and training

- **Recording.** The program runs through the normal runtime, so each `triageInbox`, `diagnoseRun` and `planNext`
  call lands in the machine's call store (`ts-host/src/calls/`). The call store keys them by definition revision.
- **Specializer.** `diagnoseRun` is repetitive: most calls are healthy runs with a recent progress line, no error
  candidates and passing gates. The specializer (`applications/specializer`) targets functions by tokens spent
  (`main.ts:33`) and groups calls by what they did. The healthy groups have structural conditions (unit active,
  `minutes_since_progress` below the threshold, `error_candidates` empty) and compile to crisp cases; failures stay
  natural language. `planNext` for a healthy cycle (no proposals) compiles the same way.
- **Training data.** A heartbeat row is the whole trajectory: evidence in, diagnosis, plan, then the agent's
  actions and the outcome label (section 7). Per `AGENTS.md` ("whole trajectories"), prompts and inputs are trained
  on, with log and tool text at lower weight. Rows enter a corpus only through
  `training/neuralese_corpora.json` with an immutable SHA-256 manifest; nothing is admitted by this program. Split by
  day, so a run's consecutive cycles never straddle train and held-out. Evidence text is `Untrusted<string>` and
  keeps that mark in the rows.
- **Harness bench.** The recorded cycles are also an evaluation set for the harness/companion work (memory: "Harness
  as killer app"): replay a recorded `Evidence` through a candidate model and compare its plan with the agent's.
- **Bootstrapping fixtures.** The incident directories under `runs/` (for example
  `neuralese-v20-checkpoint-eviction-incident-20261008-v1`) and the stall-on-refused-admission episode that
  motivated the heartbeat give the first replay fixtures for `diagnoseRun` before the program runs live.

## 10. Refinement-type candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `Reading.text`, log tails, gate text, `CoordMessage.body`, next-steps text | `Untrusted<string>` | crisp marking |
| `RunDiagnosis.evidence[].quote` | `Is<string, "occurs in the reading with reading_id">` | crisp (substring) |
| `RunDiagnosis.evidence[].reading_id` | `Is<string, "the id of a reading in run.readings">` | crisp |
| `RunDiagnosis.health`, `cause` | closed unions | crisp |
| `RunDiagnosis` | `Is<RunDiagnosis, "health is gate-failed whenever a gate field reports failure">` | crisp |
| `RunDiagnosis.summary` | `Is<string, "one sentence naming the run, its health and the cause">` | judged |
| `Obligation.message_id` | `Is<string, "the id of a message in messages">` | crisp |
| `Obligation.what` | `Is<string, "one sentence stating what the message asks of this machine">` | judged |
| `InboxTriage.digest`, `Plan.summary` | `Is<string, "at most three sentences">` | crisp (approximate) |
| `Proposal.action` | `Is<string, "an id in allowlist">` | crisp |
| `Proposal.target` | `Is<string, "a watched run id, a watched unit, a message id or an allowlisted path">` | crisp |
| `Proposal.params` | `Is<Record<string,string>, "valid against the params schema of the action">` | crisp |
| `Proposal.cites` | `Is<string[], "ids that occur in diagnoses or obligations">` | crisp |
| `Proposal.why` | `Is<string, "one sentence citing a diagnosis or obligation">` | judged |
| `Plan.idle_resources` | `Is<string, "empty when no resource is idle, otherwise names the idle resource">` | crisp against the resource numbers (a threshold setting) and judged for the name |
| `schedule-recheck.minutes` | `Is<number, "a positive whole number of minutes below the timer interval">` | crisp |

## 11. Model-facing changes needing live measurement

All three functions are new, so there is no earlier wording to compare. Measurement means replaying recorded
`Evidence` (fixtures from the incident directories, then shadow-stage cycles) and sampling each function about 48
times per variant:

1. `diagnoseRun`: health and cause agreement with the agent's recorded diagnosis, and quote validity rate.
2. `triageInbox`: obligation kind agreement with a hand labelling of the last 50 messages in `.coordination/mail/`.
3. `planNext`: proposal agreement (section 7) and the rate of proposals dropped by the applier's schema checks.
4. The positive wording of the steps (for example "pick the line that explains the stop") against a variant
   with the cause table moved into the `Cause` type text.
5. `Untrusted<string>` rendering of logs and messages (log text can contain instruction-like content). Add the
   heartbeat functions to the existing unmeasured entry in `plans/MODEL_FACING_CHANGES.md`.

## 12. Layout, when built

```
applications/heartbeat/
  natlang.json
  diagnoseRun.nl  triageInbox.nl  planNext.nl
  collect/            crisp read-only collectors (one file per source)
  apply.ts            the allowlist applier and receipts
  compare.ts          agent-action derivation and agreement metrics
  actions.json        the allowlist (owner-edited data)
  watch.json          declared runs (sessions append)
  main.ts             CLI: collect, run, report, compare
```

Report outputs: `runs/heartbeat/<date>.jsonl` (records), `runs/heartbeat/latest.md` (rendered by crisp code).

## 13. Questions for the owner

1. The auto-apply set in section 6 is small on purpose. Should `release-cache` apply automatically from stage 2, given
   that "full machine autonomy" already lets the DGX agent manage memory?
2. Process control stays a proposal. Should an "urgent" proposal (a failed run that the owning session has not
   addressed for N cycles) also post a coordination request to that session? The program would still not execute it.
3. Executor choice (`:8083`) and the degraded mode when no executor answers: confirm.
4. Who edits `watch.json`: the owning session at job start (proposed), or should the ledger's `run`/`adopt` register
   an entry automatically?
5. Promotion thresholds in section 7 (days, share, minimum cycles): the owner's numbers.
6. `AGENTS.md` names `scripts/coordination_inbox.py`; this checkout has `scripts/coord.py`. Which is canonical for
   the collector?
7. Admission of heartbeat trajectories as training data: held until the owner reviews the first corpus.
