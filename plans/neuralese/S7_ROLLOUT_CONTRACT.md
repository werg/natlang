# S7 rollout contract (task record, environment, rollout record, update round)

2026-10-09. Implements S7_RL.md §10 items 1–4 in their first form. Code: `ts-host/scripts/rl/rollout-episodes.mjs`
(driver), `training/neuralese/natlang_neuralese/rl/grpo.py` (update rounds), tests in `tests/neuralese/test_rl_grpo.py`.

## 1. Rollout task record (`natlang.rollout-task/1`)

One support case of a skill episode (`natlang.skill-episode/1`, the shared episode library). Query and transfer cases
are never rolled out: they stay sealed for evaluation (the episode gate's rule).

| Field | Meaning |
| --- | --- |
| `id` | `<episode id>#<case id>` |
| `family`, `split` | from the episode |
| `environment` | `{kind: "natlang-episode", episodes_file, episodes_sha256, episode, case, entry, export}` |
| `fixture` | `{args}`: the case's arguments |
| `check` | `{metric, expected: "episode-case"}`: the episode's metric; the expected value stays in the episode file (host-only) |
| `budget` | `{max_model_calls, timeout_ms}` |
| `difficulty` | the episode's headroom support quality for this executor, when screened |
| `provenance` | licence, source groups, case group |

## 2. Environment contract

`natlang-episode` environments reset, run and check through code that already exists:

- **reset**: a fresh `Folder` snapshot of the episode's target files and library (`skillEpisodeFiles`); nothing
  persists between rollouts.
- **run**: the native runtime executes the target's entry export on the case's arguments (`SourceEvaluator`, one case,
  usage gateway with the task budget), against an OpenAI-compatible server: the reference server, vLLM (text) or
  the llama.cpp fork. The policy is passed per request (`x_natlang_adapters`, temperature, a per-rollout seed).
- **check**: the episode's host-only scorer (`episodeScoring`, `skills/scoring.ts`), pinned by the hashes of its
  built code. Fixture and timeout failures are **unscored** (`reward: null`), never reward 0: infrastructure
  failures must not teach the policy.

## 3. Rollout record (`natlang.rollout/1`)

`{round, task_id, family, group, sample, seed, policy: {endpoint, model, adapters, temperature}, turns, reward,
unscored, model_calls, wall_ms, pins}`. `turns` are the model turns of the run, recorded from the wire:
`{messages, tools, target, finish_reason}`, with `messages`/`tools` exactly as sent and `target` the assistant message
as returned (content and tool calls). They are replayable as `logLikelihood` terms of `/v1/neuralese/grad`.
`reward` is `{quality, passed, gates}`.

## 4. Update round (`natlang.grpo-round/1`)

Group-relative policy optimisation on an adapter policy held by the model server (S7 §4.1, §5.4):

1. Behaviour log-probabilities: Σ over turns of log π under the sampling adapter (a grad request without arguments).
2. Reference log-probabilities: the same without the adapter. Reward shaping `r − β (log π − log π_ref)`.
3. Advantages per task group, `(r − mean)/(std + ε)`; groups without spread give no gradient (logged).
4. Steps: sequence importance ratios `ρ = exp(log π − log π_b)`; the clipped surrogate's gradient `A·ρ·∇log π` where
   unclipped, 0 where clipped; each turn of rollout `i` becomes a `logLikelihood` term with weight `A_i ρ_i / N`;
   `/v1/neuralese/optim` (Adam) gives the next adapter block.

Not yet: Neuralese payload log-densities as sampled actions and stop decisions beyond what `logLikelihood` already
scores (it scores written blocks' stop decisions and payload densities when the recorded turn wrote blocks), token-
level ratios, stale-round discarding, monitors (§6), the rubric runner (§7), authoring-task and resource rewards
(§3.2–3.3), multi-call graph records (S5 graph replay). Rollouts on vLLM need the server to accept the adapter (S4 §7).
