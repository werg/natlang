# Ops placement: decomposition, part by part

Placement decides where one job runs: on DGX, on Pop, in a CoreWeave serverless GPU sandbox, in the DGX memory
ledger's queue, or as a decision for the owner. Two pressures make it a policy and not a fixed rule. The DGX GB10's
121 GB unified memory is the training bottleneck, and the CoreWeave sandboxes cost money and cannot reach the
teacher (plans/COREWEAVE_SANDBOXES.md). Following AGENTS.md ("policy is natural language; mechanism is crisp"), the
decision is a natural-language function. The facts it reads, the bound on its output, and the launch are crisp.

This document follows the same pattern as applications/ops-retention/DECOMPOSITION.md, with the same decision kinds:

- **fn**: its own natural-language function.
- **inline**: one instruction inside its caller.
- **implicit**: left to the model.
- **crisp**: a helper.
- **pluggable**: a crisp and a natural-language implementation, with crisp as the default.
- **service**: the outside world.

## Policy and mechanism

- **Natural language: the decision.** Given one job's needs and the fleet's facts, `placeJob.nl` decides where the
  job runs and why.
- **Crisp: the facts, which have one right answer** (`scripts/coreweave/placement.py`):
  - DGX headroom and the queue ahead, from `memory_ledger.py status`;
  - the job's needs, from its job file (`needs`: measured GPU peak, hours, teacher, lineage owner, whether it moves an
    active lineage);
  - the CoreWeave sandboxes this machine has running, from cwctl's launch records;
  - readiness: a credential, a bucket, and a recorded passed GPU smoke;
  - the budget left this week, from the settings (`training/coreweave-placement.json`) and the cost bounds of this
    week's launches.
- **Crisp: the exact bound and the effect.** `refusal()` in index.ts refuses a natural-language decision in these
  cases:
  - it places on CoreWeave a job that does not fit one GPU, needs the teacher, exceeds the parallel limit or the
    budget, or arrives while CoreWeave is not ready;
  - it places on DGX a job that needs more than the headroom;
  - it places on Pop a job whose lineage Pop does not own;
  - it moves an active lineage without asking the owner.

  When a decision is refused, the crisp decision serves. The effect is `cwctl.py launch`, which launches only a job
  whose recorded decision is `coreweave`.
- **Teach in errors, not prompts.** The steps state the wanted decision for each case. A refused decision is
  recorded with the rule it broke, in the shadow log `~/.local/state/natlang/coreweave/placement-shadow.jsonl`.

## Data

All data types are in `types.ts`. The Python side builds the same field names.

- `JobNeeds`:
  - `id`, `kind`;
  - `gpu_memory_gb` (measured peak, else declared), `hours`;
  - `needs_teacher`, `lineage_owner`, `moves_active_lineage`;
  - `coreweave_cost_usd` (an upper bound, or null while the GPU rate is unknown).
- `FleetFacts`:
  - `dgx_headroom_gb`, `dgx_waiting_ahead`;
  - `coreweave_ready`, `coreweave_gpu_memory_gb`, `coreweave_fit_margin_gb`;
  - `coreweave_active`, `coreweave_max_parallel`, `coreweave_budget_left_usd`.
- `PlacementDecision`: `where` (`dgx | pop | coreweave | queue-dgx | ask`) and `reason`.
- `PlacementItem`: `id`, `job`, `fleet`, and `crisp` (the crisp decision).
- Shadow record: `time`, `mode`, `id`, `crisp_where`, `crisp_reason`, `nl_where`, `nl_reason`, `refused`, `error`,
  `served`.

## Parts

| Part | Decision | Unit | Why |
|---|---|---|---|
| DGX headroom and queue ahead | crisp | `dgx_facts()` | the memory ledger is the outside world; the numbers are exact |
| Job needs from the job file | crisp | `job_needs()` | field reads and the cost bound (`job.cost_estimate`): arithmetic |
| CoreWeave readiness, active count, budget left | crisp | `coreweave_facts()` | launch records and settings; counting and subtraction |
| **Decide where one job runs** | **pluggable** | crisp `crisp_decision()` (Python); nl `placeJob.nl` | the policy. One job per call keeps each call small for small executors |
| Pop-owned lineage → Pop | inline | step 1 | one comparison |
| Moving an active lineage → ask the owner | inline | step 2 | one flag; AGENTS.md keeps such decisions with the owner |
| Teacher needed → DGX (or its queue) | inline | step 3 | one flag and one comparison |
| Fits DGX now → DGX | inline | step 4 | two comparisons; DGX costs nothing extra |
| Fits a sandbox, within limits and budget → CoreWeave | inline | step 5 | four comparisons over computed facts |
| Otherwise the DGX queue, naming the blocker | inline | step 6 | the default |
| The reason text | implicit | each step | free wording; the shadow log compares `where` only |
| Request sizing (CPU, memory) | crisp, in the job file | `job.py` validation; measured peaks in `needs` | billed on the request: arithmetic from measurements, not judgment |
| Bound on an nl decision | crisp | `refusal()` in index.ts | an exact verifier of the stage's output; a launch never follows a refused decision |
| Mode setting, shadow comparison, trace event | crisp (existing) | `pluggable()` in ts-host | one implementation per concept |
| Python to natlang bridge | crisp | `placement.decide()` runs `node ts-host/scripts/placement-decide.mjs` | the same subprocess bridge as storage retention |
| Record the decision in the job file | crisp | `cwctl.py place --write` | durability |
| Launch | service | `cwctl.py launch` (CoreWeave API) | the outside world |

Why one function and not one per rule: as with retention, each rule is one or a few comparisons over facts that
crisp code has already computed. Numbered steps in one function keep the order explicit (the first rule that applies
wins), and that order is the policy's core: ownership first, then the owner's decisions, then the teacher, then the
free machine, then the paid one.

## The algorithm (as `placeJob.nl` spells it out)

1. `lineage_owner` is `pop` → `pop`.
2. `moves_active_lineage` → `ask`.
3. `needs_teacher` → `dgx` if it fits the DGX headroom, else `queue-dgx`.
4. Fits the DGX headroom and nothing of its priority waits ahead → `dgx`.
5. CoreWeave is ready, the job fits one GPU minus the margin, there is room under the parallel limit, and the budget
   allows it (no budget set, or a known cost within it) → `coreweave`.
6. Otherwise → `queue-dgx`, naming the first blocker.

## Modes and how to run them

The setting is `mode` in `training/coreweave-placement.json`. It is overridden by `NATLANG_CW_PLACEMENT`, which is
overridden by `cwctl.py place --mode`. The values are:

- `crisp` (default): no model is called.
- `shadow`: both run, and crisp serves. Disagreements, refusals and errors go to the shadow log.
- `nl`: the natural-language decision serves after the bound. If it is refused, crisp serves.

```
python3 scripts/coreweave/cwctl.py place JOB.json [--mode shadow --model <served id>] [--write]
python3 scripts/coreweave/cwctl.py launch JOB.json      # refuses unless the recorded decision is coreweave
```

Live measurement before `nl` serves: about 48 samples per variant on the live executor, inside a teacher window, as
plans/MODEL_FACING_CHANGES.md asks. The instructions are model-facing and unmeasured, so they are debt until then.

## Not yet ported

- Pop's own facts (its GPU headroom). Today Pop is chosen only for Pop-owned lineages.
- Ordering many waiting jobs against each other (which of ten arms goes first). Today each job is decided alone, and
  the parallel limit is the only interaction between them.
- Choosing the request size by judgment. It stays crisp from measured peaks.
