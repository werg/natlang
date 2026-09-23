# External datasets to natlang trajectories

> **Status (2026-09-23):** historical plan. Its implementation ran on the Python runtime, which has been removed; see [PROGRAM_IR_PIPELINE.md](PROGRAM_IR_PIPELINE.md#retired-sources) for what remains.

Status: implementation plan, 2026-09-19. This plan uses the sources in
`TRAINING.md` §3.3 and the Jev-like datasets investigated in September 2026.
It does not treat a classifier response as an interpreter trace. Dataset or
Jev labels supply semantic leaf values; natlang's reference policy supplies
tool actions and the runtime verifies them.

## Objective and boundary

Train two things from each usable source:

1. **A typed leaf**: an instruction, explicit arguments, a finite return type,
   and a checked answer. A leaf may need a `read` before `write(return, ...)`.
2. **A stated program**: pseudocode that calls leaves over records, branches on
   their results, and uses crisp helpers for exact operations. Its trajectory
   comes from `scripts/generate.py:run_program` and `ReferenceAgent`, which
   execute every action through the actual harness and check the return value.

The target is the interpreter's next tool action, not Jev's probabilities or a
chain of reasoning. Preserve Jev scores in provenance for curation and later
calibration experiments; do not put them in the student's input or target.

## Source priority

| Source | Label status | Best natlang use | Initial priority |
| --- | --- | --- | --- |
| [NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data) | Programmatic gold and saved Jev answers for most records; per-row CC0 metadata | Smart-home and support decisions, catalog lookup, contrastive rules; shared state with several typed questions | 1 |
| [jeff benchmark data](https://github.com/logan-markewich/jeff/tree/main/bench/data) | 1,600 committed labeled rows from eight upstream tasks | Topic, sentiment, spam, BoolQ and rating leaves; map/count programs | 1, subject to upstream terms |
| [LocalLLaMA typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 1,600 synthetic cases, five teacher-labeled questions each; Apache-2.0 | Customer-service, invoice, security and agent-review programs with cross-field constraints | 2; synthetic teacher labels need checking |
| [SaaS sales conversations](https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations) | 100,000 synthetic conversations; final outcome and automatic per-turn annotations; Apache-2.0 | Role-aware customer-turn judgments and prefix/fold programs | 2; Jev-label new turn tasks |
| [kev suites](https://github.com/jaredpalmer/kev/tree/main/evals) | Ready-made prompts and provenance; inspected rows have no inline gold | Policy and routing prompt pool; reconstruct upstream gold or Jev-label | 2; preserve locked tests |
| [Sajid576 SQL Injection Dataset](https://www.kaggle.com/datasets/sajid576/sql-injection-dataset) | 30,919 binary rows, with some defects; Kaggle license = Unknown | Security input triage and batch policy programs | Pilot/evaluation now; distributed training after terms are settled |
| [Nimble](https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md) | Published training JSONL is git-ignored and absent from fresh checkout | Reuse contrastive curation method and public benchmark converters | Recipe, not an import |
| [jevlike](https://github.com/vinnylarouge/jevlike) | Reproducible exact-match menus; external Wikispeedia click data | Small protocol drills; navigation only if the program states the routing problem | Low |

Use source labels as hard answers only when their semantics match the natlang
question. The label hierarchy is: independently executable/programmatic truth,
human annotation with a usable guideline, audited synthetic label, then Jev
alone. Record disagreement instead of silently replacing an upstream label.

## Canonical intermediate record

Make one source adapter per dataset. All adapters emit a common JSONL record
before constructing `Program` instances:

```json
{
  "id": "source/split/row/question",
  "source": "...",
  "source_revision": "...",
  "license": "...",
  "group_id": "...",
  "split": "train",
  "state": "text, or a small typed record",
  "question": "natural-language instruction referring to args paths",
  "args": {"item": "...", "rubric": "..."},
  "return_type": "\"a\" | \"b\" | \"unknown\"",
  "gold": "a",
  "gold_source": "programmatic | human | synthetic | jev | adjudicated",
  "jev": {"model": "...", "scores": {}},
  "source_meta": {}
}
```

Keep source ID, hash, grouping key, original split, and label provenance even
after conversion to tool trajectories. Make the train/test split at the
**source group** level before deriving multiple questions, prefixes, or
counterfactuals from a record. Never train on a source's locked test split.

## Program families to build

| Family | Pseudocode structure | Semantic leaves | Exact checks and final value |
| --- | --- | --- | --- |
| `decision_leaf` | Judge one item against an explicit rubric; return an enum or Bool | Any accepted question from NanoJev, jeff, sales, kev, or SQL data | Gold value equals the typed return; the harness checks every tool action |
| `batch_triage` | `for each item: classify`; filter, group and count; optionally escalate `unknown` | Topic, intent, objections, SQL risk | `std` helpers compute counts and preserve item alignment; compare full report |
| `multi_decision_case` | Judge several independent questions about one state, then apply a stated decision table | NanoJev and typed-decisions workflows | Crisp function applies precedence and checks cross-field consistency; compare action and explanation codes |
| `sales_prefix` | For each customer turn, classify objection, buying signal and requested next step; fold the active concerns; select follow-up action | Jev-labeled customer turns from role-tagged conversations | Prefix only sees earlier turns; fold and final routing are exact. Whole-conversation `outcome` is a separate auxiliary/evaluation label, never a per-prefix oracle |
| `security_screen` | Normalize/segment a batch, judge each query, apply allow/review/block policy | SQL injection leaves plus bounded unknown cases | Normalization, grouping and action policy are crisp; verify aggregate action and flagged indices |
| `contrastive_rule` | Run the same written policy on a minimal pair and aggregate what changed | NanoJev's programmatic rules; Nimble-style counterfactuals | One fact change must flip the checked leaf value; pair stays in one split |

For each family, make a one-leaf form, a short composed form, and a multi-item
form. Use human-readable `.nl` bodies for the composed forms; the reference
policy reads the program structure, while the student must interpret its
natural-language instructions. Paraphrase those bodies only after a round-trip
check through the harness. Include a small number of ambiguous inputs that
exercise `report_blocker` instead of forcing a label.

For example, a sales prefix program can say: “Take each customer turn so far;
classify its main objection; collect the still-open objections; select the
follow-up action from the written priority rules; return the action and the
objection counts.” One source conversation produces several **separate**
prefix programs. A checked reference trajectory calls the turn classifier as
a map, calls or runs exact aggregation, chooses the stated branch, writes the
typed report, then completes. A one-turn version has the same source text but
only the leaf's `read` (when needed), `write(return, label)`, and completion.
The final conversion outcome is unavailable to these prefix programs.

### Source-specific choices

- **NanoJev:** Begin with `smart_home_v2`, `support_decisions_v1`, and
  `catalog_lookup_v2`. Use its reference `gold` for the trajectory and compare
  saved Jev answers as a quality measure. For navigation and tic-tac-toe, put
  BFS/minimax and arithmetic in crisp helpers; use those families to teach
  delegation and control flow rather than asking the small model to solve the
  board internally. Stage 2 includes replay examples from stage 1: deduplicate
  by state/family before counting or splitting.
- **jeff / kev:** Retain the question's instructions and option descriptions.
  Labels such as `urgent` alone are weaker than an explicit decision rubric.
  jeff's benchmark JSONL has inline gold. kev's inspected suites have prompts
  and upstream row IDs but no inline gold; reconstruct from the pinned upstream
  revision or send prompt-only training candidates to Jev. Keep all benchmark
  test and locked records outside training.
- **Sales:** Parse `conversation` JSON, not `full_text`, to retain speakers.
  Exclude `scenario`, future turns, `outcome`, `probability_trajectory`, and
  other automatic scores from a prefix judgment's input. Initial Jev questions:
  primary customer objection (price / integration / training / security /
  capability / timing / none), intent (ask / evaluate / request demo / commit /
  decline / unclear), and whether the seller's last reply addressed the
  customer's stated concern. The latter needs a small manual audit. The
  3,072 embedding columns are irrelevant; stream the 7.17 GB CSV and discard
  them. Split by company/product or scenario family before taking prefixes.
- **SQL injection:** Preserve raw query bytes as text; do not execute them.
  Remove empty strings, exact conflicting-label strings, and duplicates before
  splitting. Normalize whitespace, comments, numbers and quoted literals only
  for **grouping** similar payload templates, not for changing the student's
  input. Create `unknown` cases from genuinely incomplete fragments and audit
  them separately. Jev is a second label source; it is not the sole oracle.

## Jev labeling protocol

Use classifier.dev's versioned `POST /v1/classify` on public texts only. A
request has explicit `labels`, an instruction/rubric, `tier: "fast"`, and a
batch of inputs with the same question. For multiple distinct questions over
one state, use dimensions with separate instructions if the service's current
contract supports the desired criteria; otherwise issue separate batches.
Represent Bool as two named labels and ordered scores as named levels. A
forced-choice `score` is a discrete rubric level, not a continuous target.
Jev supplies finite decisions here; record extraction and free-text drafting
need source gold, crisp code, or a different checked teacher.

Archive the full request and response, response model per item, API version,
scores, status, retry history, and a hash of each input. Keep only responses
actually attributed to Jev in the Jev-labeled pool; the gateway documents
fallback models. Run a small audited pilot before bulk labeling. Do not use a
fixed confidence cutoff as proof of correctness: scores may be rounded, and
calibration does not transfer automatically to these domains. Prefer
agreement with independent gold, counterfactual consistency, and adjudication
of disagreements. Observe the service's current per-IP classification limits.

## Implementation sequence and acceptance gates

1. **Adapters and manifest.** Add streaming adapters for NanoJev, jeff,
   typed-decisions, sales CSV, kev and SQL CSV. Write normalized records and a
   manifest with source revision, file hash, license, row counts, split method,
   and discarded-row reasons. Keep unknown-license data in a separate pool.
2. **Leaf pilot.** Build 100–200 `decision_leaf` programs each from NanoJev,
   jeff, sales and SQL data, balanced by answer and source group. Label only
   the prompt-only sales items and selected disagreements through Jev. Audit
   examples in every class, plus ambiguous and contradictory cases.
3. **Composed pilot.** Author and verify at least one `.nl` program per table
   family above. Generate batches and case variants with `run_program`; check
   that every sample is accepted by the turn grammar and reaches the expected
   final value. Inspect traces to ensure calls, loops, branches and exact
   helpers appear, rather than only leaf writes.
4. **Separation and evaluation.** Freeze source-group test splits before
   scaling. Report next-turn accuracy by action kind and whole-program success
   by family. Hold out at least one source family or domain per program shape,
   and evaluate against the original source gold where available. Do not infer
   real-world sales or security performance from synthetic/bench data alone.
5. **Scale the useful families.** Increase only families whose labels pass
   audit and whose composed programs add measurable interpreter coverage.
   Keep a stable replay slice of existing natlang data to detect regressions.

The pilot is ready to scale when its manifest reproduces exactly, the harness
verifies every retained trajectory, no group crosses a split, source/Jev
disagreements have an explicit disposition, and composed families exercise
more than a single `write(return, ...)` action.
