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
together or use distinct `--out-name episodes-NAME` values. Assemble the SFT
mixture with `sft_sources.py build`, passing `--min-open-goal-share 0.8` to
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

## Capability coverage

| Capability | Evidence in the first cases | Gap |
| --- | --- | --- |
| Discover relevant state | Agent chooses search queries and sources, or lists and reads workspace files | Browser and CDP, provider accounts, remote devices, and multi-workspace discovery |
| Choose and compose skills | Goal-only root chooses research, revision, or a revision that depends on the research result | Broad installed skill selection, branching and replanning across longer workflows |
| Execute independently verifiable work | Public evidence is read from the live cache. A workspace revision is independently read back | Publishing, deployment, integration changes, UI state, and durable cross-service outcomes |
| Recover safely | A concurrent file edit makes the agent reread, replan, and verify a revision that retains the competing edit | Service interruptions, partial commits, delegated work, and follow-up goals across sessions |
| Generalize to fresh goals | A source group ties related public tasks together | Real user-goal intake and held source groups for every capability family |
| Train from trajectories | Captured live exchange context is validated and assembled into per-call SFT targets | A live capture projection into the canonical semantic trajectory IR and measured results on held open goals |
