# P19 — Scheduling and personal workflow engine

Status: local scheduling application in `codebases/scheduler/` and
`applications/scheduling.mjs`. [Shared capabilities](README.md).

Natlang chooses among host-enumerated complete feasible schedules using a
request's soft preferences. Exact UTC-minute arithmetic checks durations,
windows, dependencies, existing commitments and non-overlap. The host then
revalidates the selected schedule against its current revision before commit.
Explicit block observations invalidate stale candidates, and duplicate event
IDs are checked. Integration tests execute the natlang planner and exercise
conflict, staleness, dependency-cycle and missing-offset cases.

The fixture is in memory and has no live calendar adapter. Candidate
enumeration reports truncation; selection quality, external conditional
writes, timer events and durable replanning remain product gates.

## Natlang prerequisites

C0/C7 suffice for finite scheduling proposals and exact constraint checks. C5 adds incremental replanning from explicit events. C1 only becomes necessary for real calendar/task objects and external updates. Timers are events from the host, not a new temporal language primitive. No in-episode interruption is needed initially.

## Programme and typed boundary

`plan.nl(tasks, availability, preferences) -> ScheduleResult` and `step.nl(state, event) -> PlannerState` call `interpret_request.nl`, `rank_tradeoffs.nl`, `propose_changes.nl` and `explain_replan.nl`. Crisp helpers check time intervals, required order, capacity and resource availability.

Records distinguish hard constraints, soft preferences, explicit unknowns and proposed commitments. Calendar/source revisions and time-zone identity accompany observed availability. Use exact date/time conversion in the host/helper; the model does not infer time-zone rules or nonexistent free time.

## Crisp environment

The first host supplies calendar snapshots and exact scheduling helpers as values/functions. A live host can directly share its calendar client and retained task objects with eval. Native objects remain there; portable observations and conditional update results cross into natlang.

External writes use expected revisions or a recheck before commit. A calendar API that cannot support atomic conditional updates needs an explicit reconciliation policy. A scheduling preference is not an authorisation to silently change unrelated commitments.

## Reduction and stream shape

Fold consumes task requests, availability changes, user decisions and timer observations. Steps propose or conditionally apply changes. A new event arriving during planning waits; a result is revalidated against current host revision before an external write.

If the task is too long for useful responsiveness, split interpretation, proposal and commit into short explicit stages with corresponding events. Evaluate that composition before asking for ambient interrupts. State can be saved between complete steps without persisting an unfinished model context.

## Delivery and checks

1. Plan a fixture workday with dependencies and preferences. Gate: all hard constraints checked; preference score reported separately.
2. Deliver an interruption event and replan. Gate: preserved commitments and explanations tied to the actual conflict set.
3. Add missing-duration, infeasible and contradictory requests. Gate: honest alternatives/unknowns instead of invented feasibility.
4. Connect a live-like adapter with stale revision and duplicate-event fixtures before using real calendars.

Test daylight-saving transitions using exact fixtures, unavailable resources, rescheduled prerequisites and timer duplication. Clock inputs are explicit observations.

## Trace and teacher

Capture availability revisions, interpreted preferences, proposed changes, exact conflicts and actual commit results. Teacher contrast pairs differ in feasibility or priority; evaluate unseen task structures rather than only paraphrases. A local fixture host needs neither network nor a durable timer service. The interactive product may add those without changing natlang's type system or function model.
