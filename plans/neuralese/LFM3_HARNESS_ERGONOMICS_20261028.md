# LFM3-1.7B harness ergonomics (2026-10-28)

Goal: before LFM3-1.7B training starts, find what in the natlang harness makes tasks needlessly hard for
`LiquidAI/LFM3-1.7B` (instruct, rev 40ebe0aa, official BF16 GGUF sha 39158a79…5e9f68), fix it in the runtime and
measure the fixes live. Branch `lfm3-harness-ergonomics` (not pushed). Owner decision 14 in
`plans/neuralese/LFM3_COMMITTED_DECISIONS_20261028.md`: one combined variant against the baseline, about 48
samples each on the dev packet; bisect only on a regression.

## Setup

- Dev packet `natlang-dev-task-packet-v2-20261028` (v1 with repaired edit_stream/contract_diagnosis oracles; the
  committed `scripts/coreweave/jobs/lfm3/dev-packet-spec.json` binds it). 48 cases: rows 0-2 of each of the 15
  executable shapes plus rows 3, 35 and 67 (workbench, multihop, greenhouse: the base-eval failure shapes); the
  supplementary reducer run adds rows 91-95 so all 8 `folder_criteria_reducer` cases are covered. edit_stream is
  refused by the runtime (source review hold). The held packet was used only for the starting diagnosis (base
  eval traces), never for selection.
- Settings as the base eval: `scripts/held_task_budget_probe.py`, greedy (temperature 0), 7,200 s / 1,000
  requests / 1,000 turns per case, reasoning preserved (llama.cpp default), XML tool calls via `--jinja`.
- Pop (RTX 4060 Laptop 8 GB): our fork `neuralese-lfm3` 85a2e7167 built with CUDA 11.5 for sm_86 (gcc-10 as CUDA
  host compiler: nvcc 11.5 cannot parse gcc-11's `<functional>`), `-np 8 -c 262144`. ~26 tok/s per slot.
  Every request was logged through a recording proxy, so the exact rendered prompts and error texts are available.
- Replays (`scripts/error_text_replay.py` method, own script for the proxy-log format): the turn after a captured
  dev-packet context is resampled at T=0.7 with the old and the new text; DGX llama.cpp `d05706b` build.

## Harness bugs found

1. **Probe provenance names the wrong model.** `held_task_budget_probe.py` never passed `--model-id`, so every
   LFM3 result (base eval and these runs) says `model: Ternary-Bonsai-2-27B`, `collection_role: teacher`. Fixed:
   `--model-id`, default the server's `/v1/models` id, recorded in `budget-probe.json`.
2. **`invalid-binding` tells an `import` to rename its result.** `import { helpdesk } from '…'` is lowered to a
   destructuring declaration and got the service message "give the result its own name, such as helpdeskResult" —
   wrong advice for an import. 9 dev contexts.
3. **`invalid-binding` for a stub of a live service recommends keeping the stub under a new name** ("as in
   const graphResult = { links: (node) => [] }"): the measured 2026-10-13 wording assumed the initializer uses the
   service.
4. **`read_code` answers "no-such-function … this program has none of its own" for names the call can see**: the
   executing call's own name (5 of 18 dev rejections: `invoice_total`, `current_roster`, `deliver`,
   `repair_schedule`), a scope input, a type alias of the signature (`Schedule`), a bare method of a scope service
   (`certificate`, `open`, `service_lift`), a source file name (`resolve.js`, `./resolve.js`).
5. **Unbound-name note is mostly irrelevant**: every `ReferenceError` got two sentences about template
   interpolation, including `repair_schedule is not defined` inside `repair_schedule`.
6. **No guard against an inline `nl` child re-asking its own question.** The prompt says not to; LFM3 does it
   anyway, nesting `nl@eval:1` → `nl@eval:1` … with the same instruction until the depth limit, repeatedly, until
   the 7,200 s case budget runs out (5 of 48 cases in both arms; see failure mode 1).
7. (Pop operations, not harness) `npm ci` on Pop fails intermittently (ENETUNREACH on the GitHub codeload
   dependency) and wipes `node_modules`; retrying works. Pop went offline at about 13:00 under memory pressure
   (96 concurrent collector processes beside llama-server on 14 GB); the confirmation runs moved to DGX.

Checked and found correct: the rendered LFM3 prompt (`/apply-template` on the actual logged requests): system
prompt plus `<tools>` JSON, host-generated opening eval rendered as `<tool_call><function=eval><parameter=code>`
with its reasoning, tool results in `<tool_response>`, history thinking kept. All 821 base-eval/dev replies were
parsed tool calls (one text-only reply, one 8,192-token max_tokens hit): the XML call format is not a problem.

## Ranked failure modes (dev baseline, 48 cases, 34 successes)

| # | Failure mode | Evidence (baseline) | Harness or training |
|---|---|---|---|
| 1 | **Runaway delegation copied from the system-prompt example.** When LFM3 uses `nl`, it copies the prompt example verbatim (`nl<boolean>\`Decide whether item meets policy.\`(item)`, `const policy = 'Refunds are allowed within 30 days…'`): 43 evals in the baseline, 86 in v1; half of the inline children take no arguments (18 of 38 `nl@eval:N()` children) and children re-delegate the same instruction | all 5 budget exhaustions (workbench 1-3, reducer 88-89) in both arms; 7,200 s each; 67-172 requests per case | harness guard (bug 6) + training (per-item delegation with arguments) |
| 2 | **Not believing the declared scope is live.** First action of 15 root calls with services: 11 `read_code` of a service whose declaration the opening eval already shows, 3 `import { svc } from '…'`, 1 direct use; stubs (`const facts = { page: … /* placeholder */ }`) and literal copies of inputs (`const reminders = [ … ]`) | `invalid-binding` 46 actions in 32 of 48 tasks; `no-such-function` 18 in 14 tasks | errors fixed (bugs 2-4); the habit is a training target |
| 3 | **Reasoning leaks into code**: `import { x } from '…'? Not sure.`, prose and ``` fences inside the `code` parameter | `typescript-syntax` 15 actions in 14 tasks | training |
| 4 | **Wrong `nl` call shape**: `nl<boolean>(\`…\`(item))`, `nl(\`…\`)`, zero-argument calls | `nl-not-tag` 17 error results (request log) | error shows the call rewritten (fix 4) |
| 5 | **Re-running effectful code**: after a successful reservation eval, rerunning the loop on the live inventory (v1 live_inventory 2 of 3 failures) | live_inventory | training (and see "not made" 3) |
| 6 | **Loops**: `while`, pagination `for (let p = 1; more; p++)` | `forbidden-loop` 9 actions in 7 tasks; errors teach the fix, always recovered | owner rule (structural termination); training |
| 7 | **Thinking length**: named calls median 1,530 reasoning characters (p90 6,015); inline children 1,738 | — | training; error-text fixes shortened the next turn (below) |

Idioms: `nl` appears in 83 baseline evals (37 in the correct tag form) — the base eval's "zero `nl`" was the held
subset; on the dev packet LFM3 delegates but badly. `iterateOn` 6, `.sort()` 2, `Promise.all` 2, `decide` 1.

### Directory reducers (`folder_criteria_reducer`, owner addition)

The dev packet's only reducer shape (8 cases). The instructions ask for one inline `nl` judgment per ticket,
then moves into `archive/`.

- **Folder/file-tool interface**: the reducer offers 15 tools (eval, file tools, `bash`, `python`, `editor`,
  `delegate`, …). LFM3 reads every file with `read_file` (48 calls over the 8 worlds), moves with `bash mv`
  (sometimes `cd /home/user` first, which does not exist), occasionally sends Python to `eval`; it never reads
  a file through `folder.file(p).readText()` in eval.
- **Binding values from the owning file**: because the text came through `read_file`, it is retyped into code
  as string literals (`nl<boolean>(\`Decide whether item meets policy.\`('Status: resolved\nThe customer replied
  …'))`) instead of binding `await folder.file(path).readText()` and passing it; the judgment loses the file
  identity and is exposed to copy errors.
- **Cursor/coverage errors**: not exercised. No dev packet shape iterates a folder with carried state; the
  semantic-iterate reducer families that do (`build-semantic-iterate-reducers-v18…v26`) use fixed training
  worlds and cannot be a dev packet without a separate decontamination. A reducer dev packet is needed (see
  "not made" 4).
- Outcomes: base 1/3 (rows 88-90) with two budget exhaustions from failure mode 1; supplementary rows 91-95 in
  the combined runs below.

## Changes (branch `lfm3-harness-ergonomics`)

| Commit | Change | Measured |
|---|---|---|
| 24ff89e8f | probe records the served model id | provenance only |
| 74bca872d | `read_code` of own name / scope value / file name / type alias / service member answers what the name is; ReferenceError of own name says it is the executing call, template advice only with interpolated code; `invalid-binding` for imports ("already in this eval's scope; use it directly instead of importing it") and live-service stubs ("already this scope's live X service; call its methods directly instead of defining a replacement") | replay, 18 dev contexts × 8 samples = 144 per variant: next action makes progress 78 → 101, repeats the mistake 24 → 12, median reasoning 1,290 → 874 chars. By kind: import repeat 20/72 → 11/72; service member progress 12/16 → 16/16; own name repeat 4/40 → 1/40 |
| be2f446f1 | `nl-not-tag` shows the model's own call rewritten in the tag form | in the combined variant only |
| d6b779f5d | skills: recovery advice for each new answer/error | — |
| e6ab51e67 | service member lookup: only declared functions are described as callable (v1 told `read_code("Inventory")` to call `inventory.Inventory(...)`) | bug found in v1 |
| 1a3bba49b | an ad hoc `nl` call with the same instructions as an enclosing ad hoc call is refused: "this call was itself created with these same instructions; make the judgment here from its inputs and return it instead of delegating it again" | in the combined variant only |

Measured and **not adopted**: an opening-eval result sentence "X and Y are live in this eval scope: call them
directly in eval." Replay of the first turn, 17 dev contexts × 8 = 136 per variant: direct service use 9 → 21,
`read_code` of the already shown service 96 → 84, but imports 14 → 18; mixed, and it would change the opening of
every admitted trajectory with services. Reverted.

### Task-level results

COMBINED_RESULTS_PLACEHOLDER

## Changes worth making that were not made

1. **System-prompt `nl` examples are copied verbatim** (failure mode 1): `Decide whether item meets policy.`,
   the refund policy, `isFinished`. Replacing them with examples whose names cannot be mistaken for the task's
   (or fewer examples) is the largest remaining lever, but it changes the system prompt of every admitted
   trajectory and needs its own ~48-sample measurement. Logged as debt.
2. **Zero-argument inline children** (`nl\`…\`()` over nothing in scope): an error when an inline call gets no
   arguments and its instructions mention no bound name would stop the empty children; needs a careful rule for
   captured context, so not added blind.
3. **Successful effectful evals do not list their effects.** Failures report "Already performed (not undone)";
   a success that called a live object's mutating methods says only "Stored local …", and LFM3 reran the
   reservations. A one-line effect receipt on success is model-facing on every effectful trajectory.
4. **Reducer dev packet.** Build a protected dev packet with folder-iterating reducers (cursor, coverage,
   carried state) from worlds no training build uses; until then reducer cursor/coverage behaviour is unmeasured.
5. **`read_file` result does not say how to bind the file in eval** (reducer binding failure). A footer such as
   `(in eval: folder.file("tickets/blom.md"))` would teach binding from the owning file; it touches every
   file-tool trajectory.
6. **Reducer tool surface**: `bash`/`python`/`editor` beside the Folder API invites `mv` and Python; narrowing
   the offered tools for directory reducers is a policy decision for the owner of the file tools.

## What training should cover rather than the harness

- Calling declared services and functions directly in the first eval; never re-declaring inputs or stubbing
  services (the errors now teach it in one turn, but the turn is wasted).
- Per-item delegation with arguments: `await Promise.all(items.map(item => nl<boolean>\`…\`(item)))` with the
  task's own criterion, never the prompt example; answering inside a typed child instead of re-delegating.
- Exact work in `eval`: filters, sorts and counts in code (base-eval cases 0 and 16); binding file contents with
  the Folder API in reducers and moving with `moveTo`.
- Code hygiene: no reasoning text or fences inside `code`.
- Effect awareness: do not rerun an effectful eval that already succeeded.
- `iterateOn` for open-ended loops instead of `while` / unbounded `for`.

## Training-data consistency (re-render flags)

The new texts change what the runtime renders after these errors; existing admitted trajectories keep the old
texts. Affected spans: `read_code` rejections `no-such-function, expected a function of this program…` for the
cases listed in bug 4 (now an answer or a longer rejection); `invalid-binding` messages for imports and service
stubs; the unbound-name note of `ReferenceError` (template-advice sentence now only with interpolated code, and a
new sentence for the executing call's own name); `nl-not-tag` for a function-call form with a template argument;
the new nested same-question refusal. Conversion should re-render these turns through the current runtime when
it replays admitted trajectories. The opening eval, system prompt and tool schemas are unchanged.

## Artifacts

- Pop (offline since ~13:00): `~/lfm3-eval/{base,v1,base-red,v1-red,base2,v2}`, request logs `*.requests.jsonl`.
  Local copies of the baseline and v1 results and logs in the session scratchpad.
- DGX: `/home/werg/data/lfm3-eval/ergo-20261028/{base3,v3}` (combined confirmation runs, request logs).
