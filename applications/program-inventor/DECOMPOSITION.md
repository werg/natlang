# Program inventor (`applications/program-inventor/`): Decomposition

Design: `plans/NOVEL_PROGRAMS_20261124.md`. The process that replaces procedural task generation: a frontier model
reads an atlas of every program idea natlang already has, invents a completely new one, and the idea becomes one
complete idiomatic program with two or three literal input instances and an exact checker. Every requirement it must
meet is in `requirements.json`, crisp or judged.

The process is a natlang program. Its judgments (what to invent, whether it is new, whether a program meets a judged
requirement, who is right in a disagreement) are `.nl` functions on frontier models. Its exact parts (the atlas,
nearest-neighbour retrieval, the crisp requirements, packaging, replay) and its contacts with the outside world (the
repository, the registry, the runtime, a coding agent that writes and runs files, teacher models) are services. The
orchestration is TypeScript (`invent.ts`).

## Parts

| # | Part | Form | Why |
|---|---|---|---|
| 1 | Build the atlas: one line per grpo-pool family (train rows), application, design brief, example, dataset source and accepted invented idea | service `atlas()` over `scripts/rl_tasks/program_atlas.py build` | exact; reads the registry, the repository and the accepted ledger. Rebuilt at every attempt, so it is current by construction and includes the round's own accepted ideas |
| 2 | Invent one idea that differs from every atlas entry and every idea rejected this round | fn `inventIdea.nl` (`model: inventor`) | the creative judgment the whole process exists for |
| 3 | Exact checks of the idea card: slug free, ≥ 3 judgments with different questions, ≥ 3 constructs, `nearest` names real atlas ids | crisp `ideaProblems` (invent.ts) | exact facts; a failure goes back as a rejected idea with its reason |
| 4 | Retrieve the existing ideas nearest the new one | service `near()` (`program_atlas.py near`: TF-IDF cosine, top 12, plus every accepted idea) | exact retrieval; it decides only which entries the judge reads in full, never whether the idea is new |
| 5 | Is the idea new: differs from each near entry in ≥ 2 of user/job, core judgments, computational shape, input kind | fn `isNew.nl` (`readout: decision`, `model: judge`, another family than the inventor) | judgment; scored decision. Cross-family so one model's taste does not judge itself |
| 6 | Why not (for the next attempt) | inline `nl` in `attempt` | a one-sentence reason the inventor reads in `rejected` |
| 7 | Build the idea into `program/`, `submission.json` (instances, references, ablations, wrong answers, keyword baselines) and `NOTES.md` | service `build()`: a frontier coding agent (Claude Code `claude -p`, or Codex `codex exec`) given `BUILD_BRIEF.md` | writing a multi-file program needs a shell to run `natlang check` and the qualifier; that is a coding-agent harness, the outside world. The natlang-native harness for this is `applications/pi` (pi's coding agent ported to natlang); it replaces the CLI agents when it runs frontier models end to end |
| 8 | Crisp qualification: F1 `natlang check`; F2-F4, S1-S3, C1, C2, D1, D2, D4, N2 by static checks and replay | service `qualify()` over `scripts/rl_tasks/sources/invented.py qualify` (static checks, packaging to `natlang.program/2`, `verify.mjs`, `ablate.mjs`) | exact; the collector's own replay path judges references, wrong answers, keyword baselines and judgment ablations |
| 9 | Repair until qualification passes | `semanticRepair` from `@natlang/node/std` with `check` = part 8 and `fix` = service `revise()` (the same agent with the problem list), 3 attempts | std convergence combinator: exact check, judged fix |
| 10 | Judged requirements (S4, S5, C3, D3, D5, N1) | not in the round: a review of a random sample of admitted programs (`invented.py sample-reviews`, `REVIEW_BRIEF.md`) grades them, lists expected answers a careful reader rejects (those programs are withheld from publication) and feeds its findings into `BUILD_BRIEF.md` and the skill | owner 2026-11-25: "reviews should only be on a drawn sample"; qualification admits, so the quality is set at the top of the funnel (the brief), not by per-program review loops |
| 11 | Live run: a teacher model executes the root on each instance through the collector | service `liveRun()` over `teacher_gen.py --pool CASES` | the outside world (teacher providers, the collector); it proves a real run reaches the checked answer and yields the first teacher episodes |
| 12 | Settle a disagreement between the expected answer and a teacher's answer | fn `adjudicate.nl` (`readout: decision`: `expected-wrong` / `answer-wrong` / `ambiguous`) | judgment; `expected-wrong` and `ambiguous` send the submission back, `answer-wrong` stands |
| 13 | Record the accepted idea in the ledger the atlas reads | service `record()` (append to `OUT/accepted.jsonl`) | exact durability |
| 14 | A round: attempts one after another, each seeing every idea accepted before it | crisp `inventRound` (counted `for` over a bound read once) | sequencing; sequential so the atlas grows inside the round |

Admission to training is not part of the program: the round's packaged cases (`qualification/cases.jsonl` of each
accepted submission) are registered and promoted by the data pipeline (`data_registry.py record`, `held_eval verify`,
`measure_repetition.py`, `data_registry.py promote --role grpo-pool`), a deliberate step outside the round.

## Models

| Runtime model | Role | Default |
|---|---|---|
| `inventor` | part 2 | GPT-6.1 Sol (`openai-codex`) and Claude Opus 5.5, alternating by round so neither family's taste fills the atlas |
| `judge` | parts 5, 6 | the other family than the round's inventor |
| `reviewer` | part 12 | the other family than the round's builder |
| builder (service) | parts 7, 9 | Claude Opus 5.5 (Claude Code) or GPT-6.1 Sol (Codex) |
| teacher (service) | part 11 | `ops/ei/teachers.json`'s Luna, then Sol on a failure |

Profiles named `inventor`, `judge` and `reviewer` in `~/.config/natlang/config.json` select them.

## Not built yet

- `applications/pi` as the builder backend (part 7) in place of the CLI agents.
- A natural-language atlas card per entry (`describeIdea.nl`, cached by content hash): the crisp line (family name,
  shapes, first sentence of one root instruction) is thin for authored families whose instructions open with data
  descriptions.
- Program-writing tasks with an execution checker (the student writes the program from the idea card; a fixed executor
  runs it on the held instances): only the SFT form of the authoring calls exists (design §7).
