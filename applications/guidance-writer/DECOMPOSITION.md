# Hindsight hint writer: decomposition

`writeHint` turns one verified successful execution τ* of a training task into a short hint that a self-teacher run of
the same task imports as Neuralese (plans/neuralese/NEURALESE_HINDSIGHT_HINTS.md). It is a data-processing natlang
program: τ* is an ordinary input value that the writer explores with eval. τ* never appears in full in the writer's
context. Its only full appearance is as the self-teacher's teacher-forced target.

| Part | Form | Why |
| --- | --- | --- |
| τ* as a value (`TrajectoryCall`: a call tree; each node has its function, prompt, own steps, result and children) | crisp data, in the runtime's own transcript shape (`TrajectoryStep` = `TranscriptEntry`) | The writer explores it as it explores its own `transcript`. The runtime lists it like any large argument, with `<<cut off …>>` and the full value left in the eval scope. |
| Which calls of τ* get a writer pass | crisp rule on a recorded parameter (`rl/hindsight.py` `node_classes`): the root always; a child call whose own assistant-action tokens reach `child_hint_min_tokens` gets an independent pass over its own subtrajectory; a shorter child is trained by plain SFT on its own trajectory with its actual inputs | Size threshold, not judgment: below it a useful hint would be about as long as the child's own actions. |
| Structural overview (`trajectoryOverview`) | crisp helper | Exact counting (own steps, turns, tools, statuses, sizes, errors, and children's sizes) gives a cold-start map of where to look. It is mechanism, not judgment. |
| Which facts a node's writer sees | crisp rule `visibility.ts` / `privileged_context.applicable` (existing) | A child node sees only facts matched to its own call; the root sees world-level facts. |
| Accompanying facts | existing `selectGuidanceFacts` (NL pluggable) over the world's visible facts, without the trajectory itself | The same selection as `writeGuidance`. |
| Which parts of τ* matter, and reading them | policy: `writeHint` steps 3 to 4, through eval | Which steps decided the outcome is a judgment. Reading is ordinary eval code the writer writes. |
| The hint | policy: `writeHint` result, `Neuralese<string>`; the configured writer re-encodes the returned text as the producer span | A natural-language result. Its state is what the consumer imports. |
| How short, and what not to copy | policy text (step 5), plus training: the rate penalty and the reconstruction reward (crisp math, `rl/hindsight.py`) | The model learns the trade-off from the reward. Instructions only state the intent. |
| Custody of τ* and facts | crisp verifier (`common.custody.ProtectedWorlds`, `rl/hindsight.py`) | Train worlds only; fails closed. |

`writeHint` runs once per hinted call: the root, and each long child as an independent pass over its own subtree
value. It takes no children's hints and does not recurse; short children get no hint (plain SFT).

Training: exploration actions and hint tokens are the writer episode's sampled actions and receive the score-function
term. The hint span state receives the exact reconstruction gradient through the consumer import. One packed graph
holds the writer episode, its producer span and τ*'s consumer turns.

Model-facing text (`writeHint.nl`, the overview field names) is unmeasured debt in plans/MODEL_FACING_CHANGES.md
until it is sampled live; the pilot is its first live sample.
