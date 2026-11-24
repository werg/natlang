# New ideas, built as complete programs: the process that replaces procedural task generation

**Created:** 2026-11-24. **Status:** designed, piloted (§12); nothing admitted to `grpo-pool` yet (owner review first).
**Program:** `applications/program-inventor/` (DECOMPOSITION.md first). **Crisp mechanism:**
`scripts/rl_tasks/program_atlas.py`, `scripts/rl_tasks/sources/invented.py`, `scripts/rl_tasks/ablate.mjs`.

## 1. The owner's request

> "I also want us to design a process for generating truly new ideas and building truly complex, new, native and
> idiomatic, functional semantic programming NatLang programs. That hopefully would replace the fixed idea procedural
> generation. The approach will be to provide an overview of the worlds and ideas and tasks that already exist, and
> always have the model come up with a completely new idea. And there should be clear requirements on functional,
> semantic, complexity, and code-based depth."

Earlier the same day: "I want to rip out task worlds from our generation process. That's not helpful. No procedural
generation." (DECISIONS "No invented task worlds".)

What it replaces: authoring rounds 1-23 and sfp1-3 wrote a *design theme* per family, then a generator script (`gen.py`)
that emitted 6-20 programs x 2-30 worlds by substitution; the family, shape, construct and domain were assigned from a
rotation (`rotation.jsonl`). Two measured defects follow from that shape: the variants share one idea (one judgment,
one call graph, one input kind), and the oracle selected cue-word desks (plans/semantic-fp/README.md §1: "exact match +
literal blind-solve selected cue-word desks"). The new process has no generator, no rotation, no family of variants:
**one idea, one program, two or three hand-written instances.**

## 2. Overview of what exists: the atlas

The inventor reads one document, the atlas (`program_atlas.py build`), with one line per existing idea:

| Group | Source | Line |
|---|---|---|
| Ideas this process already accepted | `OUT/accepted.jsonl` (the inventor's ledger) | slug, pitch, judgments |
| Applications | `applications/*/natlang.json` | name and description |
| Examples | `examples/*` | name and description |
| Design-brief programs | `codebases/design_briefs/*/brief.json` | title and representative episode |
| Authored task families in training | `grpo-pool` role, `auth_*` families, train rows | family, row count, top shapes, first sentence of one root instruction |
| Other task families in training | `grpo-pool` role, other families | the same |
| Dataset sources | `scripts/rl_tasks/SOURCES` | name and converter |

**Current:** the atlas is rebuilt from those sources at the start of every attempt (seconds), so it always reflects the
registry role, the repository and the round's own accepted ideas. Nothing is cached across rounds except the ledger.
Held data is not in it: pool lines come from train rows only, and held-v3/v4 tasks are not in any listed source.

**Compact:** one line per idea, at most 220 characters, grouped. The pilot atlas (§12) had 740 entries,
120 kB (~30k tokens): fits every frontier inventor's context with room for the instructions. A family counts once
however many rows it has (the 28,730-row memetic source and the 941-row codebase repair family are one line each), so
row volume does not crowd out ideas. When it outgrows a context window, the first step is a natural-language card per
entry (`describeIdea.nl`, cached by content hash) that merges families with one idea into one line; not built (the
crisp lines are thin for authored families whose root instructions open with a data description, §13).

## 3. "Completely new": definition, enforcement, check

**Definition.** An idea is new when it differs from every existing idea in at least two of four respects: (1) who runs
it and for what repeated job; (2) the judgments of meaning at its core; (3) the shape of its computation (how exact
steps and judgments feed each other, what state is carried, where results combine); (4) the kind of input data. A
change of subject matter (another industry, other names or numbers) is no difference. This is what the old rounds
called "renaming and toggling", made into a positive test.

**Enforcement, in order:**
1. The inventor (`inventIdea.nl`) is asked for exactly that, reads the whole atlas, and must name its nearest atlas
   entries by id and state the difference in the four respects. No topic, domain, construct or seed is given to it:
   there is nothing to rotate through.
2. Crisp: the slug is free, the nearest ids exist (`ideaProblems`).
3. Retrieval (`program_atlas.py near`): the 12 atlas entries with the highest TF-IDF cosine to the idea card, plus every
   idea the process already accepted. This is a floor on what the judge sees, not a threshold that decides.
4. `isNew.nl`, a scored decision by a model of the **other family** than the inventor, compares the card with each
   retrieved entry in the four respects. A false comes back to the inventor as a rejected idea with a one-sentence
   reason; the next attempt in the round sees it.
5. After the build, `measure_repetition.py` over the round's cases and the pool reports exact-structure and
   instruction repetition (`program structure` keys), and the round report lists the top retrieval similarity of every
   accepted idea. Both are recorded per round, so a drift toward a crowded region shows as a number (§10).

## 4. Requirements (`applications/program-inventor/requirements.json`)

Each is checkable: **crisp** ones by `invented.py qualify` (static analysis of the files, `natlang check`, replay through
the collector's own code), **judged** ones by `meetsBar.nl` (a scored decision by the other model family).

| Id | Group | Check | Requirement |
|---|---|---|---|
| F1 | functional | crisp | complete package; `natlang check --json` reports no errors |
| F2 | functional | crisp | 2-3 literal instances, each with an exact checker (exact/normalized `expected`, or a trusted verifier) |
| F3 | functional | crisp | each instance's reference is a root eval that runs the program's own helpers with scripted judgment answers; its returned value is accepted |
| F4 | functional | crisp | >= 3 wrong answers per instance, each from a named real mistake, all rejected |
| F5 | functional | crisp | a teacher model running the root with the real runtime is accepted on >= 1 instance; a disagreement is adjudicated (`adjudicate.nl`) |
| S1 | semantic | crisp | >= 3 distinct judgments of meaning (`.nl` files or inline `nl`) |
| S2 | semantic | crisp | **the answer depends on every judgment**: replaying the reference with one judgment's answers set to its trap reading is rejected (`ablate.mjs`) |
| S3 | semantic | crisp | a keyword baseline (root script, no `nl` call) per instance runs to an answer and is rejected |
| S4 | semantic | judged | each judgment is one narrow question with one defensible answer per item; outside text is `Untrusted`; each judgment receives what it needs and nothing it must not see |
| C1 | complexity | crisp | >= 5 functions (`.nl` + exported TS); call chain >= 3 deep |
| C2 | complexity | crisp | each instance >= 8 items or >= 1,200 words |
| C3 | complexity | judged | a judgment reads another judgment's result (or an exact stage built from judgments); >= 2 rules or stages whose order or precedence decides the outcome |
| D1 | code depth | crisp | >= 40 lines of exact TypeScript logic in the callable folder that the answer depends on |
| D2 | code depth | crisp | >= 3 semantic-FP constructs (decide/readout, std combinator, higher-order judgment, iterateOn...until, structural recursion, ensures, Is/Untrusted, Neuralese<T>, folder transaction, pluggable), found in the source |
| D3 | code depth | judged | every exact step is TypeScript; `.nl` instructions state judgments and name helpers, no algorithm in prose |
| D4 | code depth | crisp | the root returns a named, doc-commented type from `types.ts`; no output-contract paragraph |
| N1 | novelty | judged | `isNew` (§3) |
| N2 | novelty | crisp | no generator: instances are literal JSON in `submission.json`, at most three; no script in the submission outside `program/` |

S2 is new and is the main answer to the cue-word desks: the old rounds checked that wrong *answers* were rejected; S2
checks that the *program's* answer changes when any one judgment is read wrong, through the program's real exact code.
S3 makes the keyword baseline mandatory for every instance, not only for verifier tasks.

## 5. From idea to program, instances and checker

1. **Idea card** (`IdeaCard` in `types.ts`): slug, pitch, user, >= 3 judgments (question, result type, trap), shape,
   the exact work, constructs, input description, output and how it is checked, why keywords fail, nearest ids and the
   difference.
2. **Build** (`BUILD_BRIEF.md`, the build service): a frontier coding agent in an empty submission directory writes
   - `program/`: `natlang.json`, `types.ts` (types first, doc comments on fields), the root `.nl` (the operation the
     student executes in training), its callable folder with one `.nl` per judgment (`readout: decision` for finite
     answers) and TypeScript helpers for every exact step (which may import sibling `.nl` judgments and
     `@natlang/node/std`);
   - `submission.json` (`natlang.invented-program/1`): 2-3 instances, each with inputs, `expected` (or a verifier
     binding), a reference root script plus scripted children, one ablation per judgment, >= 3 wrong answers with
     their mistakes, a keyword baseline;
   - `NOTES.md`: each judgment's trap and why each item has one reading.
   The agent runs `natlang check` and `invented.py qualify` itself (through the memory ledger) and fixes what they
   report before it returns.
3. **Repair loop** (`semanticRepair`, 3 attempts): the round re-qualifies and returns the problem list to the same
   agent.
4. **Packaging** (`invented.py package`): one `natlang.program/2` case per instance, family `inv_<slug>`, world
   `invented:<slug>` (all instances of an idea share one split), reference/wrong answers/keyword baselines/ablations
   in `curriculum`, the program files in `semantics.files`: exactly the format the collector, `verify.mjs`,
   `teacher_gen.py` and `sft_sources.py` read today.

`expected` is written by running the program literally (each judgment as the function receiving only its own
arguments would read it), which the reference replay then confirms through the real exact code.

## 6. Qualification

Per submission, `invented.py qualify` (one ledger unit, CPU only, ~1-3 min):
static crisp requirements -> `natlang check --json` -> package -> `verify.mjs` (reference replay F3, wrong answers F4,
keyword baselines S3, alias acceptance measured) -> `ablate.mjs` (S2) -> `qualification/report.json`.
Then the judged requirements (`meetsBar`, S4/C3/D3), the live run (F5: `teacher_gen.py --pool CASES`, Luna, Sol on a
failure; `adjudicate` on a disagreement).

Per round, before any admission: `held_eval verify` against held-v3 and held-v4 (custody), `measure_repetition.py`
against the pool (§3.5), registration (`data_registry.py record`).

## 7. How outputs enter training

1. **Tasks for collection and teachers.** The cases go into a registry dataset (`natlang-rl-invented-<date>`), are
   verified against held, and are promoted to `grpo-pool`. Expert iteration then treats them like any task: they start
   `unmeasured`, collection tries them on the student, never-solved ones go to the teacher tier, and every verified
   success (own first, then teacher) enters the per-task pool weighted by the measured pass rate. The live-run episodes
   of §6 are already the first teacher episodes for them.
2. **Judgment-level tasks from the reference.** Every scripted child of a reference is a judgment call with a known
   right answer in its real rendered context (the program's own call); decisions among them can be scored with the
   O3 decision objective. These come from the same programs at no extra authoring cost and are not procedural: each
   is a distinct item of a hand-written instance. (Built as a follow-up: an extractor over `curriculum.reference.children`.)
3. **The programs themselves as program-writing data.** The authoring calls are natlang calls: `inventIdea(atlas,
   rejected) -> IdeaCard` and, once the builder is a natlang program (`applications/pi`), the build calls. Each
   accepted submission is a verified teacher demonstration of those calls (plans/semantic-fp/05 §4: "idiomatic program
   demonstrations at the author level"), trained as teacher-tier SFT rows with teacher reasoning as zero-weight
   context. Until the builder runs as natlang calls, the accepted `(IdeaCard -> program files)` pairs are recorded
   in the submission directory for that conversion. A program-writing *task* with an execution checker (the student
   writes the program; a fixed executor runs it on held instances against the verifier) is the next step and is not
   built.
4. **Splits.** The world draw is 70% train / 15% dev / 15% confirmation per idea (`invented.split_of`), so a share of
   new ideas is never trained on and can measure transfer to unseen programs. Whether those join held-v4 is an owner
   decision (§14).

## 8. Which models generate

| Role | Model | Why |
|---|---|---|
| Inventor (`inventIdea`) | GPT-6.1 Sol (Codex subscription) and Claude Opus 5.5 (Claude Code), alternating by round | the strongest models we can run at no marginal cost; alternating families keeps one model's taste from filling the atlas |
| Novelty judge (`isNew`) | the other family than the round's inventor | self-judging is the failure mode of a novelty gate |
| Bar reviewer, adjudicator (`meetsBar`, `adjudicate`) | the other family than the builder | the reviewer must not grade its own program |
| Builder | Claude Opus 5.5 (Claude Code) or GPT-6.1 Sol (`codex exec`) | needs a coding agent that runs `natlang check` and the qualifier and fixes what they say |
| Live run | `ops/ei/teachers.json` teachers: Luna, then Sol on a failure | the same teachers and transport that fill the teacher tier |
| Not used | Haiku (credits out), free models (low yield on 2026-11-24's qualification; too weak to author level-3 programs), Vertex (paused) | |

## 9. Cost and throughput

Measured in the pilot (§12), per program:

| Stage | Model | Tokens | Wall time |
|---|---|---|---|
| Atlas build | crisp | 0 | ~1 min (reads 23 pool files, 38k rows) |
| Invent | GPT-6.1 Sol, high effort | ~37-39k (30k of it the atlas) | 2-4 min |
| Novelty judge | Claude Opus 5.5 / Sol | ~10-25k per idea | < 1 min |
| Build + qualify | Claude Opus 5.5 (Claude Code) | ~195-260k | 19-27 min, of which several minutes in the ledger queue |
| Judged requirements | Sol | ~15-20k | 1-2 min |
| Live run | Luna, Sol on failure | 300-5,000 completion tokens per episode (the prompts dominate) | 1-4 min per episode |

All models used run on subscriptions (Claude Code, Codex): the marginal money cost is zero and the binding limits are the
subscriptions' rate windows and the DGX ledger's queue (every `natlang check`, qualification and live run is a ledger
unit; on 2026-11-24 evening a 2-4 GB unit waited 5-12 min for admission). Six builders in parallel produced six
qualified programs (12 tasks) in ~30 min wall. A standing process of 4-6 builders is therefore ~100-200 new programs
(200-400 level-3 tasks, each its own idea) per day if the subscription windows allow it, against ~38k pool rows today:
the value is breadth of ideas and depth per task, not volume. Collection cost per task is that of any long level-3
task (16-40 judgment calls per episode).

## 10. How it avoids drifting back into fixed ideas or procedural generation

- **No generator can exist:** N2 rejects any script in the submission outside `program/`, and instances are literal
  JSON, at most three per idea. Families are ideas: `inv_<slug>` has 2-3 rows.
- **No rotation, no assigned theme:** the inventor gets the atlas and the rejected list, nothing else. Coverage comes
  from the atlas showing what is crowded, not from a cell list.
- **The atlas includes the process's own output**, rebuilt every attempt, so the process cannot repeat itself, and the
  novelty judge sees every accepted idea, not just the 12 nearest.
- **Cross-family judging** (inventor and judge from different families).
- **S2 and S3 for every instance**, so an idea whose answer a cue rule reaches, or whose judgments do not matter,
  cannot pass however it is dressed.
- **Numbers per round:** accepted / attempted, `isNew` rejection rate, top retrieval similarity of accepted ideas,
  `measure_repetition` structure repetition, constructs used. A rising similarity or structure repetition is the
  drift signal, read at every round.

## 11. Rules that bind it

AGENTS.md "Task supply" (exact oracles, qualified by reference, negative and alias replay; repetition measured before
admission; never pad by repetition), "Data" (registry, roles, `held_eval verify`), "Natlang-native work" (the process is
a natlang program; DECOMPOSITION.md first), the owner's 2026-11-24 rule (no task worlds, no procedural generation).

## 12. Pilot (2026-11-24)

Data: `/home/werg/data/natlang-novel-programs/pilot-20261124/` (`atlas.md`, `ideas/`, `ideas-r2/`, `judge/`, `sub/<slug>/`
with `program/`, `submission.json`, `NOTES.md`, `qualification/`; `bar/`, `bar2/`, `live1/`, `live2/`). The stages ran
as the program's functions dispatched by hand (the `.nl` instructions rendered verbatim into `codex exec` and Claude
Code subagent calls), with the crisp services run for real through the memory ledger; `natlang run
applications/program-inventor` itself was not run end to end (§13).

**Round 1: GPT-6.1 Sol invents, Claude Opus 5.5 judges and builds.** Six sequential `inventIdea` calls, each seeing the
atlas plus the ideas before it. All six judged new by Opus (`isNew` probabilities 0.82-0.92; closest entries: design
briefs 009, 030, 041, 068, 071 and the measurement harmonizer). Six Opus builders in parallel; **all six passed crisp
qualification** (F1-F4, S1-S3, C1, C2, D1, D2, D4, N2):

| Program | Functions (.nl) | TS logic lines | Constructs found | Instance words / items | Wrong answers | Ablations rejected |
|---|---|---|---|---|---|---|
| `causal-adjustment-auditor` | 19 (5) | 178 | decide, ensures, higher-order, refinement, std | 1068 / 16; 893 / 18 | 10 | 8/8 |
| `nonvisual-interaction-parity-auditor` | 11 (4) | 208 | + iterate-until | 2060 / 31; 1716 / 27 | 10 | 6/6 |
| `suppressed-count-reconstruction-auditor` | 20 (5) | 229 | decide, ensures, higher-order, refinement, std | 1234 / 20; 847 / 14 | 10 | 8/8 |
| `assurance-case-defeater-grounder` | 21 (5) | 186 | decide, ensures, iterate-until, refinement, std | 1265 / 16; 1052 / 14 | 10 | 8/8 |
| `anatomical-character-tree-scorer` | 21 (5) | 226 | decide, ensures, higher-order, refinement, std | 782 / 13; 636 / 13 | 8 | 8/8 |
| `randomization-law-reconstructor` | 19 (6) | 418 | + iterate-until | 1391 / 20; 1199 / 16 | 8 | 10/10 |

Every keyword baseline ran to an answer and was rejected; every reference replayed through the program's own
TypeScript to the expected answer. Builders needed 1-3 qualification runs.

**Judged requirements (Sol as reviewer).** First pass: C3 met by all four reviewed, D3 met by three (the parity
auditor's root walks its tasks in prose), S4 failed by all four, every time on the same clause, "every value from outside
is typed `Untrusted`": the programs wrap outside free text but not ids, numbers and closed labels. That wording was
mine and asked for more than the skill's purpose. S4 now reads "every free text from outside that a judgment reads",
and all six were re-reviewed: S4 still failed for all six (each left some outside free text a judgment reads, such as
labels, definitions or titles, as plain `string`; the parity auditor also withheld data a judgment needed), D3 failed
for two (roots that spell out the per-item walk in prose), C3 passed for all six. **The repair step then ran as
designed**: each builder got the reviewer's findings, fixed them (types made `Untrusted`, a judgment given the records
it needed, root walks moved into TypeScript helpers), and requalified with unchanged answers. Third review: five of six
meet every judged requirement; `suppressed-count-reconstruction-auditor` keeps one S4 finding (a marked-cell display
string built in TypeScript reaches a judgment untyped).

**Live run (F5).** `teacher_gen.py` over the 12 cases, Luna first, Sol on a Luna failure. Two findings came before
any verdict:
- **The 40-call episode budget makes these programs unsolvable by construction.** The first run gave
  `budget_exhaustion` at 39-40 turns: the programs make 16-40 judgment calls per instance, and every child call counts
  against the shared `EPISODE_MODEL_CALLS` (40). Rerun with `--max-model-requests 400`.
- **Codex's subscription usage limit hit mid-run** ("You have hit your ChatGPT usage limit") on nested child calls. The
  root then returned `failed` or a wrong value and `teacher_gen` scored a teacher failure (`semantic_mismatch`), not a
  provider failure. Fixed on main (d670ae655: such an episode is no verdict, deferred, teacher cools down; test). The run
  was stopped once the window was spent, since Luna and Sol share it with the production teacher campaign.

Clean verdicts (no limit in the episode's logs): **Sol solved `randomization-law-reconstructor` a and b and
`assurance-case-defeater-grounder` b** (23-39 turns, ~1 min each). Clean failures: `causal-adjustment-auditor` b
(Luna and Sol), `nonvisual-interaction-parity-auditor` b (Luna and Sol), `causal-adjustment-auditor` a and
`assurance-case-defeater-grounder` a (Luna; Sol not reached cleanly). Not run before the stop:
`suppressed-count-reconstruction-auditor`, `anatomical-character-tree-scorer`, parity auditor a. So F5 is met for 2 of
6 programs and open for 4. Adjudication of the two double failures (`adjudicate.nl` by
Claude Opus 5.5, `judge/adjudications.json`): **`causal-adjustment-auditor` b is ambiguous**: three investigator notes are
worded as associations ("heavy users are far less likely to cancel"), and both "asserted" (expected) and "unspecified"
(Sol) are defensible under `readCausalAssumption`'s instruction; the instance must be reworded (and instance a checked
for the same wording). **The parity auditor b is answer-wrong**: Sol labelled two completion states `wrong-outcome`
where the literal reading is `pending`, and with its labels the exact stages give exactly its answer (4 surviving pairs
against 7). So the live run found one real defect in a qualified instance that every crisp check had passed: what S4's
review and F5 exist for. The adjudicator was the builder's family because the Codex window was spent (§13).

**Round 2 (check of a defect round 1 showed).** Both the Opus judge and the builders saw that round 1's six ideas share
one template: an expert's audit before an approval; 3-5 "which supplied canonical X does this passage mean" judgments
turn ~2,000 words of closed-world prose into a formal structure; then a textbook exact algorithm (d-separation, a
simulation fixed point, integer range projection, tree DP, grounded semantics, absorbing-chain probabilities). Each idea
is new against the atlas, but the *round* repeats itself. Fix (in `inventIdea.nl` now): the accepted ideas are read as
one group, and a new idea lies outside their shared pattern in at least two of user, role of the judgments and kind of
exact computation. Three ideas from Claude Opus 5.5 (the other family) with that instruction: a knitting-pattern stitch
count verifier (errata, multi-size expansion), a liturgical ordo compiler (precedence and transfer rules over a
calendar), a bug-bounty duplicate and payout ledger. Sol judged all three new (0.84-0.94) and noted the remaining
shared pattern is "specialists using meaning judgments to construct state for exact computation": the paradigm itself,
which the requirements impose (§14, decision 3). Not built.

**My judgement.** The programs meet the bar on code depth and semantic structure, clearly above the authored rounds:
judgments are narrow typed decisions composed with `semanticMap`/`semanticFilter`, `.partial` and `ensures:`, the exact
work is real TypeScript (graph search, branch and bound, exact fractions), and S2 shows that every judgment moves the
answer through the real code. Their weak points, all reported by their own builders:
1. **The instances are smaller than the ideas** (13-31 items, 640-2,060 words against cards of ~2,000 words): the
   builders shrank them to keep `expected` computable and the ablations clean.
2. **Easy items dominate**: 16-40 judgment calls per episode, of which 2-5 are hard (the traps); the rest restate the
   text. An exact checker over the whole output still fails a run on any one slip, so these are hard tasks for the
   student (whole-run exact match over dozens of calls), and the frontier will mostly send them to teachers first.
3. **The roots are thin**: the root `.nl` calls one or two TypeScript helpers that orchestrate everything, so the
   student's own root call has little to decide; the learning is in the judgment calls.
4. **One template per inventor family** (round 2 above).
5. **F5 is met for only 2 of 6** so far: strong teachers fail most of these whole-output exact checks, which is
   either wrong `expected` values (adjudication above) or the difficulty noted in 2.

## 13. What is not built or not checked

- **`natlang run applications/program-inventor` end to end.** The program passes `natlang check` and its crisp
  services ran for real, but the pilot dispatched the judgments by hand (verbatim instructions to `codex exec` and
  Claude Code subagents) and the build service as Claude Code subagents rather than through `main.ts`'s `claude -p` /
  `codex exec` service. The `liveRun` service reads `teacher_gen`'s `manifest.jsonl` for pass/fail but does not yet
  extract the teacher's returned value from the rollout for `adjudicate` (it passes the rollout paths instead).
- **The builder as a natlang program** (`applications/pi` running a frontier model), and with it the build calls as
  natlang calls for program-writing SFT (§7.3).
- **Judgment-level tasks from reference children** (§7.2): an extractor over `curriculum.reference.children`.
- **A program-writing task with an execution checker** (§7.3).
- **A natural-language atlas card per entry** (`describeIdea.nl`): the crisp lines of many `auth_*` families start
  with a data description ("draft is an earlier routing of tickets...") rather than the idea.
- **Construct detection** finds structural recursion only in `.nl` front matter; TypeScript self-recursion (two pilot
  programs use it) is not counted. D2 passed without it.

What I did not check:
- **Held custody beyond identity.** The ids and source groups are new by construction (`invented:<slug>`), so
  `held_eval verify` by id/group cannot overlap and was not run; no content-level comparison against held-v3/v4 text
  was run (the builders were told never to read held or evaluation data and invented all content).
- **The judged requirements' reliability**: one reviewer call per program, no second reviewer. The adjudicator was Opus
  (the builders' family), not the other family, because Codex's window was spent.
- **F5 for 3 programs** (suppressed-count, character-tree scorer, parity auditor a) and the ambiguous causal instance's
  repair: not run. The ambiguous instance was not fixed.
- **Whether the student can learn from these tasks**: no collection on the student, no pass rate, no SFT.
- **Correctness of `expected` beyond the replay**: the replay proves the reference reaches `expected` through the
  program's code given the scripted judgment answers; that the scripted answers are the right readings rests on the
  builder, the live run (F5) and S4's review. No independent blind solve was run apart from the live run.
- **Repetition against the pool.** `measure_repetition.py` over the twelve pilot cases alone: 6 distinct programs, 6
  distinct structures and 6 instruction texts (effective 6.0 each, top structure share 0.17). It was not run together
  with the pool, so whether a pilot structure key equals a pool one is unmeasured (unlikely: 4-6 `.nl` files plus 3-7
  TypeScript helpers per program).

## 14. Open decisions for the owner

1. **Admission.** Promote the pilot's qualified programs to `grpo-pool` (12 tasks; two of the six ideas drew dev /
   confirmation splits), or wait for a first unattended round of 20-50 ideas.
2. **Held split for new ideas.** The world draw keeps 30% of ideas out of training (dev/confirmation). Should those
   ideas extend held-v4 as a "new programs" stratum (transfer to unseen ideas), or stay as a separate measurement?
3. **The paradigm bias of the requirements.** Requirements S1-S3, C1 and D1 make every idea "judgments of meaning feed
   exact TypeScript", and the inventors converge on "expert audit: prose to formal structure to textbook algorithm".
   Other natlang shapes (stateful event loops and apps, folder transactions with verified edits, iterate-until-judged
   convergence, writing with property checks, Neuralese compaction of long inputs) are reachable only if some
   requirements become shape-specific (for example D1's 40 lines relaxed for a convergence program whose depth is in
   the loop). Keep one bar for every idea, or allow shape-specific bars?
4. **Task size against learnability.** The pilot tasks are 16-40 judgment calls with one exact check over the whole
   output. Keep them whole (hard tasks, teacher-first), or also cut each into per-stage tasks with intermediate checks?
5. **Episode budget.** The collector's 40-call episode budget (a budget stop scores 0) cannot fit a program that makes
   16-40 judgment calls per run. Options: the budget scales with the program (a declared `budget:` on the root, or
   calls counted per function rather than per episode), or the inventor's requirements cap calls per instance. The
   first is the "no premature limits" reading; it changes collection for every task, so it is listed here.
6. **Inventor families.** Sol and Opus alternate in the design; Argon (Vertex) when credits return?
