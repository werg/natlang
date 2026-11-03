# Dataset catalog for RL task authoring (agentic + knowledge reasoning), 2026-11-03

Purpose: impulses for authoring NEW natlang task programs (`natlang.rl-authored-task/1`, see the kit examples in
`/mnt/external/natlang-rl-authoring/examples/tasks.jsonl` and the rotations `rotation/constructs.txt`,
`rotation/problems.txt`). A dataset here is raw material for a world, its inputs and an exact oracle. It is not a
template to wrap. The owner rejects "generic dataset-answer wrappers" (`nl@eval(item)` "do what item.request asks")
and renaming as diversity. Each idea below names the public operations, the carried state, the call graph and the
oracle.

How this catalog was built. Every path comes from a registry, a manifest or a receipt:
`training/neuralese_corpora.json`, `training/corpus-manifests/`,
`/home/werg/data/external-rl-sources/SURVEY.md` + `<name>/PROVENANCE.json`, `acquisition.json` receipts under
`data/neuralese/external-expansion-20261014/Dnn/`, `scripts/neuralese_data/acquired_agents.py`
(`FULL_TRAJECTORY_CORPORA`) and the live RL source receipt. Schemas were read from the first record of one file.
**Workers: do the same.** Open one known file and read its head; never `find`/`du`/`grep -r` over `/mnt/external`
or `/home/werg/data`.

## Rules every task must keep

- **Excluded inputs.** These never seed, inspire the content of, or share a world with a task:
  - anything in the ProtectedWorlds union (`training/protected-worlds.json`: evaluation-only packets in the corpus
    registry, held-union and quality source groups, decision holds, `training/source-split-reservations.json`,
    LFM3 joint-v4 reserved worlds);
  - `runs/natlang-dev-task-packet-v2-20261028`;
  - `natlang-rl-broad-external-dev-20261031` / `-confirmation-20261031`;
  - the held dev/confirmation halves of the live source (`.../full-core-effects-.../held/`);
  - every upstream evaluation split: dev/validation/test of HotpotQA, MuSiQue, 2Wiki, SQuAD, DROP, QuALITY,
    TriviaQA, HybridQA, Qasper (dev+test), WTQ, TAT-QA, FinQA, MultiHiertt, ConvFinQA dev, LogiQA, bAbI, CLUTRR,
    GSM8K, MATH, MBPP, StrategyQA test, TabFact test, WikiSQL/MSR-SQA test;
  - Spider and BIRD dev/test databases (`bird-dev-*`, Spider `test_database`);
  - MATH-500 (decontamination list only);
  - the tau2 retail/airline tasks (revised tau-bench v1 *test* tasks), all tau2 `banking_knowledge` tasks and
    tau2 telecom `test`;
  - held tau users;
  - the protected MuSiQue seed groups (`/home/werg/data/broad-rl-tasks-20261031/protected-musique-groups.txt`;
    about 92% of MuSiQue train shares a seed with them).

  Use train splits only, and run the custody check before admission.
- **Visible facts, real oracles.** Every fact the answer needs must be in the call arguments: documents, tables,
  databases behind a service, tool results. Acceptable oracles:
  - an upstream gold label whose evidence is in the inputs;
  - an executed result (SQL result set, an executed program, executed tool functions);
  - a final-state digest of a service store;
  - a unit-test pass.

  Never ask for facts the program cannot see; source-reported "success" is not an oracle. Negative controls
  (meaningful wrong answers) must be rejected (`scripts/rl_tasks/verify.mjs` practice).
- **Shared oracle kinds** (`ts-host/src/evaluation/answer-sets.ts`, `puzzle-solutions.ts`):

  | kind | normalization or use |
  |---|---|
  | `exact` | MCQ letters, records |
  | `normalized` | `result-set`, `numeric-tolerance`, `answer-list`, `qa-alias-map` |
  | `span` | `qa` normalization; golds of 8 words or fewer |
  | `puzzle-solution` | countdown/24, n-queens, sokoban |
  | `authoring` | executable code with tests |
  | service-certified receipts | tool calls |
  | store `finish()` | final-state digest |

- **Underuse matters.** The live RL task source is
  `/home/werg/data/broad-rl-tasks-20261031/full-core-effects-20261102-20261102T011031Z-3015925/` (receipt:
  61,399 train cases, held dev 5,834 / confirmation 5,720). Its per-source row counts are in the table below
  ("live RL"). Almost all of its external rows are **single-call** root functions: one question, one answer.
  Even a "used" dataset is therefore open for multi-operation, carried-state and deep-call-graph programs
  (constructs `call_depth_3`, `neuralese_fan_in`, `iterate_neuralese`, `folder_reducer`, `decision_readout`,
  `multi_operation_codebase`, `effect_state`). **Prefer the rows marked `0` (unused).**
  - The 2026-11-03 mix review (`RECIPE_REVIEW_20261103/DATA_MIX_DIVERSITY.md` §4, §6.9) found the RL supply
    narrow: 94% from one generator, 5% Neuralese, and no coding, research or acquired-world tasks.

## Catalog table

Abbreviations: Ag = agentic, KR = knowledge reasoning. Paths: `ERS` = `/home/werg/data/external-rl-sources`,
`EXP` = `/home/werg/natlang/data/neuralese/external-expansion-20261014` (symlink to
`/mnt/external/natlang-development-data/...`), `BRT` = `/home/werg/data/broad-rl-tasks-20261031`.

| # | dataset | class | exact local path | records | license | split / protection | live RL |
|---|---|---|---|---|---|---|---|
| 1 | tau-bench v1 retail/airline store worlds | Ag | `BRT/assets-v3/stores/tau-{retail,airline}-db.json`; converted cases `BRT/build-v3/tau-{retail,airline}/` (raw-cases 598 / 46); domain tools in `ERS/tau2_bench/repo/src/tau2/domains/{retail,airline}/tools.py` | retail DB 500 users / 1,000 orders / 50 products | MIT | train tasks only; held tau users + all tau2 retail/airline tasks excluded | **0** (built in v3, not in live source) |
| 2 | APIGen-MT-5k | Ag | converted `BRT/build-v3/apigen-mt-{retail,airline}/raw-cases.jsonl` (1,500 / 1,285); raw archive path `agentic-20260927/apigen-mt-5k` under `/mnt/external/sdkb-archive/raw` is **absent** on DGX today | 5k trajectories | CC-BY-NC-4.0 | trajectories touching held tau users dropped | **0** |
| 3 | tau2-bench telecom | Ag | `ERS/tau2_bench/repo/data/tau2/domains/telecom/` (`tasks_full.json`, `split_tasks.json`, `db.toml`, `user_db.toml`, `tech_support_manual.md`) | 2,285 full; split train 74 / test 40 / small 20 | MIT | **train ids only**; test is a public leaderboard eval | **0** |
| 4 | NESTFUL | Ag | `ERS/nestful/data__train.jsonl` | 1,861 | Apache-2.0 | one split (benchmark); 56 gold/execution disagreements dropped in v3 | 1,264 |
| 5 | ToolACE | Ag | converted `BRT/build-v3/toolace/`, `BRT/toolace-full-schema-successor-20261031/` (raw archive absent) | ~11k | Apache-2.0 | 6% carve held | 1,500 |
| 6 | xLAM function-calling 60k | Ag | converted in live source (raw archive absent) | 60k | CC-BY-4.0 | 6% carve held; 32 tool sets shared with held dropped | 1,468 |
| 7 | Dolci-Instruct-SFT-Tool-Use (D30) | Ag | `EXP/D30/data/train-0000{0..5}-of-00006.parquet` (`messages[{role,content,functions,function_calls}]`, `dataset_source`, `id`) | 227,579 | ODC-BY (AllenAI; acquisition "accepted") | upstream decontaminated against BFCLv3; SFT-registered source | 0 |
| 8 | Finding the Right Fit (D06) | Ag | `EXP/D06/results/task_level.parquet` (harness, benchmark, model, task_id, reward, model_calls, tokens, agent_seconds, cost_usd); `trajectories/<harness>/<agent>/<model>.jsonl.gz` | 6,204 task-level rows | CC-BY-NC-4.0 | none (results table) | 0 |
| 9 | Thoughtworks agentic-coding trajectories (D40) | Ag | `EXP/D40/sessions.parquet` (session_id, source_dataset, messages_json, n_turns, ground_truth_meta_json) | 15,000 (SWE-smith ×2 / SWE-rebench 5k each) | derivative-multi-source | SFT-admitted (160k PH docs) | 0 |
| 10 | TerminalTraj (D33) | Ag | `EXP/D33/data/train-0000{0,1}-of-00002.parquet` (query, messages) | 20,000 | not stated on card | SFT evidence-terminal | 0 |
| 11 | OpenThoughts-Agent-v1-SFT (D29) | Ag | `EXP/D29/data/train-00000-of-00001.parquet` (conversations, agent, model, task, episode) | 15,209 | Apache-2.0 | SFT evidence-terminal | 0 |
| 12 | SearchSwarm-SFT (D02) | Ag/KR | `EXP/D02/train.parquet` (source, question, answer, messages, subagents, num_subagents) | 6,732 (redsearcher 6,588 / openseeker 144) | MIT | SFT Search/OWL | 0 |
| 13 | DeepDive (D31) | KR/Ag | `EXP/D31/data/{qa_rl,qa_sft,trajectories_sft}-00000-of-00001.parquet` (id, question, answer, conversations) | qa_rl 2,234 / qa_sft 1,016 / traj 858 | Apache-2.0 | none known | 0 |
| 14 | Evidence subagent SFT (D10) | Ag/KR | `EXP/D10/data/train.parquet` (id, conversations, tools, metadata: source, task_type, delegated_subtask) | train 10,379; eval 100 | not stated on card | `eval*` parquet = held; WebWalkerQA-derived | 0 |
| 15 | OWL-SFT (D21) | Ag/KR | `EXP/D21/data/train-00000-of-00001.parquet` (conversations, task_id, question) | 1,564 | Apache-2.0 | **task_ids are HotpotQA ids**: drop any in HotpotQA validation | 0 |
| 16 | CognitiveKernel-Pro SFT (D34) | Ag | `EXP/D34/{ck-pro-web,webwalker_subset,docbench,tablebench}.sft.jsonl` (messages) | 506/325/21/23 MB | CK-Pro license | `tablebench`/`docbench` derive from eval benchmarks: **do not seed** | 0 |
| 17 | MAST-Data (D16) + MAScope (D18) | Ag | `EXP/D16/MAD_{full,human_labelled}_dataset.json`; `EXP/D18/{MAST-clean,LG-Ctrl,Inject-Mix}.jsonl`, `lofo_splits.json` | 200 MB / 132 MB traces with failure labels | CC-BY-4.0 | MAScope LOFO test folds: keep held | 0 |
| 18 | CooperBench (D08, D12, D15) | Ag | `EXP/D08/data/*.parquet` (92 leader/follower), `EXP/D12/cmp-*.tar.gz`, `EXP/D15/<repo>_task/` | 92 / 6 run variants / 720 task files | MIT / Apache-2.0 | SFT-admitted | 0 |
| 19 | Acquired full sessions D13–D38 | Ag | roots per each `acquisition.json` via `acquired_agents.load_acquisition(ref)` (D22 Open-SWE, D23 SWE-zero, D24 MiroVerse, D25 SWE-rebench, D26 Toucan, D27 ToolMind, D28 Nemotron-Agentic, D32 OpenResearcher, D35 Nemotron-terminal, D36 CoderForge, D37 Orchard, D38 UltraData); D13–D20 roots are not under `EXP` today | D32 ≈85k sessions in SFT | per receipt (owner: all accepted) | 2,090 acquired rows withdrawn as protection; run the union | 0 |
| 20 | verifiable-coding-python | Ag | `ERS/verifiable_coding_python/data__train.jsonl` (problem_statement, gold_standard_solution, verification_info.test_cases stdin/stdout or fn) | 35,735 | not stated (APPS/TACO/CodeContests-derived) | may overlap APPS/CodeContests test: dedupe | 0 |
| 21 | Spider + BIRD train databases | Ag/KR | `runs/natlang-rl-broad-external-20261031/assets/sql/` (204 spider-*, 52 bird-*) | 256 SQLite DBs | CC-BY-SA-4.0 | **dev/test DBs excluded** (`bird-dev-*`, Spider test DBs) | 2,679 (single query each) |
| 22 | DataBench tables | KR | `runs/.../assets/tables/databench-NNN_*.json`; source `ERS/databench/files/data/<NNN>/{all,sample,qa}.parquet` | 65 tables, 1,830 QA | MIT | SemEval test answers not here; 6% carve | 1,826 |
| 23 | MultiHiertt | KR | `ERS/multihiertt/train.jsonl` (paragraphs, tables[html], table_description, qa{question,answer,program,table_evidence}) | train 7,830 | MIT | test has no answers; dev held | 944 |
| 24 | ConvFinQA | KR | `ERS/convfinqa/files/train_turn.json` (pre_text, post_text, table, per-turn programs, exe_ans) | 11,104 turns | MIT | dev = eval | 1,112 |
| 25 | TAT-QA / FinQA | KR | `ERS/tatqa/train.jsonl` (table, paragraphs, questions[answer, answer_type, scale, derivation]); `ERS/finqa/train.jsonl` | 2,201 ctx (~13k q) / 6,251 | MIT | dev+test held | 1,460 / 1,433 |
| 26 | HybridQA | KR | `ERS/hybridqa/train.jsonl` + `WikiTables-WithLinks-*/{tables_tok,request_tok}/<table_id>.json` | 62,682 | CC-BY-4.0 | dev held; 30 shared tables dropped | 1,469 |
| 27 | WTQ / MSR-SQA / WikiSQL / TabFact | KR | `ERS/wikitablequestions/train.jsonl`, `ERS/msr_sqa/default__train.jsonl`, `ERS/wikisql/default__train.jsonl`, `ERS/tabfact/tab_fact__train.jsonl` | 14,152 / 12,276 / 56,355 / ~92k | CC-BY-SA / MS-PL / BSD-3 / CC-BY-4.0 | tests held | ~1,500 each |
| 28 | TableInstruct (TableBench train) | KR | `ERS/tableinstruct_tablebench/files/TableInstruct_instructions.jsonl` (qtype, qsubtype, table{columns,data}, question, answer) | 19,661 | Apache-2.0 | TableBench eval itself not downloaded; dedupe tables against it if added | **0** |
| 29 | Qasper | KR | `ERS/qasper/train.jsonl` (title, abstract, full_text[sections], figures_and_tables, qas[answers: yes_no / extractive_spans / unanswerable / free_form]) | 888 papers | CC-BY-4.0 | **dev + test held** (SCROLLS/LongBench) | **0** |
| 30 | StrategyQA | KR | `ERS/strategyqa/data__train.jsonl` (question, answer bool, facts, decomposition, term) | 1,603 | MIT | test held | **0** |
| 31 | MuSiQue / 2WikiMultihopQA | KR | `ERS/musique/train.jsonl` (20 paragraphs, question_decomposition, answer, aliases, answerable); `ERS/2wikimultihopqa/train.jsonl` (evidences triples, type) | 39,876 / 167,454 | CC-BY-4.0 / Apache-2.0 | validation held; MuSiQue protected seeds | 1,278 / 1,498 |
| 32 | QuALITY / TriviaQA rc / DROP | KR | `ERS/quality/`, `ERS/triviaqa_rc_wikipedia/rc.wikipedia__train.jsonl`, `ERS/drop/` | 2,523 / 61,888 / 77,400 | CC-BY-4.0 / Apache-2.0 / CC-BY-SA-4.0 | validation held | 1,200 / 1,200 / 1,500 |
| 33 | bAbI / CLUTRR / LogiQA 2.0 | KR | `ERS/babi/train.jsonl`, `ERS/clutrr/<config>__train.jsonl`, `ERS/logiqa2/train.jsonl` | 200k / ~70k / 12,567 | CC-BY-3.0 / CC-BY-NC-4.0 / CC-BY-NC-SA-4.0 | tests held | 1,400 / 1,000 / 1,193 |
| 34 | Enigmata puzzles | KR | `ERS/enigmata/files/<family>/en/train.jsonl` (36 families: zebra_logic, crypto_KPA, sudoku, hitori, star_battle, minesweeper, campsite, magic_square, ...) | 217,541 | CC-BY-NC-4.0 | Enigmata-Eval not downloaded | **0** (needs per-family verifier port) |
| 35 | RLVR-IF constraints | KR | `ERS/rlvr_if_constraints/data__train.jsonl` (messages, ground_truth, dataset, constraint_type, constraint) | 29,946 | ODC-BY | IFEval-style (different prompts) | **0** |
| 36 | WebInstruct-verified / NuminaMath-1.5 / DeepScaleR / ORZ-57k | KR | `ERS/webinstruct_verified/data__train.jsonl` (question, answer, answer_type, category), `ERS/numinamath_1.5/data__train.jsonl`, `ERS/deepscaler/files/deepscaler.json`, `ERS/orz_math_57k/` | 228,736 / 896,215 / 40,315 / 56,878 | Apache-2.0 / Apache-2.0 / MIT / MIT | **decontaminate vs MATH-500/AIME/AMC** first; WebInstruct test held | **0** |

Not available on DGX today: the configured raw archive `/mnt/external/sdkb-archive/raw` (`common.paths` root
`archive`; checked 2026-11-03) is absent. Its items are KodCode, SynSQL-2.5M, SynLogic, ScienceWorld, ALFWorld,
TextWorld, HotpotQA raw, raw APIGen-MT/ToolACE/xLAM and the tau-bench v1 envs. Use their converted forms (rows 1,
2, 5, 6, 21) or the tau2 repo, and do not search for them.

## Program ideas per dataset

Notation: `root(args) -> T` is the public typed `.nl` function. `helper/` are callable-folder children. `N<...>` means
a `Neuralese<...>` result. "carry" is the state threaded through `iterateOn` or a service store.

**1. tau-bench store worlds** (best agentic gap: built, verified, unused)
- `resolve_ticket_queue(tickets: string[]) -> string` over a `store` service (tools ported in v3).
  - iterateOn the tickets, carrying a `N<string>` case log;
  - per ticket, `classify_request` → `plan_actions` (crisp lookups `find_user_id_by_name_zip`, `get_order_details`)
    → `apply` mutations; a `policy_gate` helper decides refuse/escalate from the policy text, which is an input.
  - Oracle: `store.finish()` final-DB digest against the digest after replaying the gold actions of the same
    (train-user) tasks.
- `audit_exchanges(user_id) -> {order_id; issue}[]`: read every order of a user and flag policy violations (e.g.
  exchange on a non-delivered order). Per-order `nl` judgement, crisp filtering. Oracle: exact list computed from the
  DB fields by the author's reference.
- Use only train-split users. Author fresh multi-request customers (a cancellation, then an address change, then a
  correction) so the digest depends on order and corrections.

**2. APIGen-MT-5k**
- `serve_conversation(turns: string[]) -> string` (store digest).
  - Customer turns arrive one per iterateOn step, with a carried `N<string>` state of what was promised and verified.
  - A `verify_identity` helper must succeed before any mutation (decision readout).
  - Oracle: final-state digest from the gold trajectory's actions executed on a fresh store.
- `summarize_effects(turns) -> {tool: string; args: Record<string,string>}[]`: plan the exact mutation list without
  executing it. Oracle: answer-list of the gold write calls (reads excluded).

**3. tau2 telecom (train ids)**
- `diagnose_line(ticket, device_state) -> string[]`.
  - `read_manual` → `N<string>` notes from `tech_support_manual.md` sections (fan-in over sections);
  - `next_step` loops over device-state readings with iterateOn until connectivity holds.
  - Oracle: the ordered user-side action list (`evaluation_criteria.actions`) plus the `env_assertions` evaluated on a
    simulated device record (port `assert_mobile_data_status`, `assert_internet_speed` as crisp checks).
- Skip tasks whose success needs `communicate_info` or NL assertions only.

**4. NESTFUL**
- `run_calculation_plan(question, tools) -> number`: the root writes a plan, and a `bind_args` helper resolves
  `$var_k.result$` references while a crisp executor runs the provided arithmetic functions. Oracle: the
  executed-chain result equals `gold_answer` (the rows where they agree).
- `repair_plan(question, broken_plan) -> plan`: plant one wrong argument binding and let a `find_fault` helper locate
  it. Oracle: the repaired plan executes to gold, and the plan diff touches only the planted step.

**5–6. ToolACE / xLAM** (already single-call in RL; go multi-op)
- `fulfil_requests(requests: string[], tools) -> receipt[]`: a session of 3–5 user requests over one tool set. A
  carried `N<string>` notes earlier results (a later request says "same city as before"). Oracle: service receipts
  of the gold calls per request, with the shared default-coercion rule. Facts must be visible in earlier requests.

**7. Dolci tool-use (D30)**
- `choose_and_call(request, functions) -> {name; arguments}[] | "none"`, where decision readout covers "no function
  applies" (the mix contains no-call rows). Oracle: exact call list after JSON canonicalization. Take only rows
  whose arguments are literally inferable from the request text, checked by the author.
- `schema_lint(functions) -> {fn; problem}[]`: per-function `nl` review plus crisp JSON-schema checks (missing
  required, enum/type mismatch) on planted defects. Oracle: the planted-defect list.

**8. Finding the Right Fit (D06)**
- `harness_report(table) -> {harness; model; reward_per_dollar}[]`: crisp aggregation over the results table
  service. An `nl` `explain_gap` helper produces `N<string>` notes that feed a `recommend` decision (which
  harness/model for a budget). Oracle: exact aggregates; the recommendation is fixed by a stated rule over those
  numbers ("highest mean reward under cost X").
- `find_anomalies(table) -> task_id[]`: rows with reward 1 but zero model calls, or cost inconsistent with tokens
  under stated prices. Oracle: exact set.

**9. D40 / D25 / D22 coding sessions** (multi_operation_codebase)
- `triage_session(messages) -> {files_touched: string[]; tests_run: string[]; final_status}`: iterateOn over turns
  with a carried `N<string>`, extracting crisp facts from tool observations. Oracle: exact sets parsed from the
  session's own structured tool calls (author computes them crisply). This is reading, not repo execution.
- `localize_failure(test_output, diff) -> {file; line}`: inputs are a recorded failing test log plus the patch.
  Oracle: the hunk named by the traceback, present in both inputs.

**10–11. TerminalTraj (D33) / OpenThoughts-Agent (D29)**
- `reconstruct_state(commands: string[], observations: string[]) -> Record<path, "created"|"deleted"|"modified">`.
  - A per-command `nl` effect reader with a carried state (effect_state).
  - Oracle: exact map computed from shell-semantics replay on a mock filesystem for the restricted command subset
    (`mkdir`, `touch`, `rm`, `mv`, `cp`, `>`, `>>`).
  - Use only sessions entirely in that subset.
- `next_command_check(goal, history, candidate) -> "advances"|"redundant"|"harmful"` where "redundant" means an
  exact repeat of an earlier command with an unchanged observation. Oracle: crisp rule.

**12–15. SearchSwarm (D02), DeepDive (D31), evidence subagent (D10), OWL (D21)** (research)
- `answer_from_dossier(question, pages: Folder) -> string`. Pages are the tool observations recorded in the
  session.
  - Per-page `read_page` → `N<string>`, fanned in by `connect_clues` (neuralese_fan_in), then `final`.
  - Oracle: span-qa against `answer`. Admit only if the gold string occurs in some observation; facts must be
    visible.
- DeepDive questions are KG random walks with obfuscated hops. Write a `decompose` helper that emits the hop chain,
  then one `resolve_hop` child per hop (call_depth_3). Oracle as above.
- `delegate_plan(question) -> {subtask: string}[]` (SearchSwarm `subagents`): judge coverage, not text. The oracle
  checks that each recorded sub-answer entity appears in some planned subtask (exact entity match).
- D21: drop any HotpotQA validation id. D10: never use its `eval*` parquet.

**16. CK-Pro web (D34)**
- `navigate_site(start_page, pages: Record<url,string>, goal) -> url`: a static site graph taken from recorded
  webwalker observations. iterateOn link-following with a carried visited set and a stop predicate. Oracle: the
  goal url (exact) whose page contains the answer. Do not use `tablebench`/`docbench` files.

**17. MAST / MAScope (D16, D18)** (diagnosis-from-evidence)
- `postmortem(trace_messages) -> {failure_modes: string[]; first_bad_step: number}`.
  - Per-agent `nl` reviewers emit `N<string>` notes; the root fans them in and does a decision readout over the
    published MAST taxonomy, which is given as input.
  - Oracle: human labels in `MAD_human_labelled_dataset.json` (answer-list).
  - `first_bad_step` only where an annotation provides it (MAScope Inject-Mix carries injected step ids).
- `topology_of(trace) -> "chain"|"star"|"debate"|...` from LG-Ctrl. Oracle: exact label. Avoid label regurgitation
  (the 2026-11-03 review withdrew formulaic SFT views); require the evidence path.

**18. CooperBench (D08/D12/D15)**
- `merge_coordination(agent_a_log, agent_b_log, comm_log) -> {conflict_files: string[]; who_yields}`.
  - Two `nl` readers produce `N<string>` notes; the root reconciles them (multi-source-reconciliation).
  - Oracle: file overlap computed from the two recorded patches; `who_yields` by a stated rule (smaller diff yields).

**19. Acquired sessions D13–D38** (Toucan/ToolMind/Nemotron-Agentic tool sessions; OpenResearcher)
- `replay_tool_session(user_goal, tool_specs, recorded_results) -> final_answer`: tool results come back from a
  recorded-response service keyed by exact call. The model must reproduce a call sequence the service can answer
  (calls it cannot answer return an error). Oracle: the final answer equals the session's final answer **and** that
  answer is derivable from the returned results (author check).
- `ledger_from_session(session) -> {tool; calls; failures}[]`: crisp counting after per-turn `nl` classification of
  failures. Oracle: exact counts parsed from structured tool results.

**20. verifiable-coding-python**
- `solve_with_helpers(statement, examples) -> code`: a root that first calls a `spec_notes` helper (`N<string>`
  constraints/edge cases), then writes code. Oracle: `authoring` with the dataset's stdin/stdout tests (needs the
  Python/IO runner the broad plan lists as missing). Dedupe against APPS/CodeContests test first.
- `explain_failing_case(code, failing_input, expected) -> line`: plant a one-line bug in `gold_standard_solution`.
  Oracle: the planted line number, and the fixed code passes.

**21. Spider/BIRD train DBs** (live RL has only one-query cases)
- `quarterly_close(db, requests: string[]) -> rows[]`: a request queue whose later requests depend on earlier
  results ("for the top region from before, list ..."). A carried `N<string>` query plan; crisp `db.query`. Oracle:
  result-set per request from gold SQL authored by the worker and executed at build time.
- `reconcile_tables(db, claims: string[]) -> {claim; holds: boolean}[]`: per-claim `nl` plan → SQL → readout.
  Claims are built from executed values (true) and perturbed values (false). Oracle: exact booleans.
- `apply_corrections(db, messages) -> digest`: a writable copy with UPDATE/INSERT effects applied in order,
  corrections replacing earlier figures (effect_state). Oracle: final table digest.

**22. DataBench tables**
- `data_brief(table, questions[5]) -> answers`: one table, several typed questions answered by shared helper
  functions (`column_profile` → `N<string>`, reused across questions; folder_reducer). Oracle: DataBench typed
  answers per question. Integers above 2^53 are fine since 66a9b1374.

**23–25. MultiHiertt / ConvFinQA / TAT-QA / FinQA** (financial reports)
- `analyst_followups(report, questions: string[]) -> number[]`.
  - A ConvFinQA dialogue as a sequence: iterateOn with a carried `N<string>` of earlier answers and resolved
    references ("that year", "the difference").
  - The arithmetic is crisp code; a `locate_cells` helper reads the html/hierarchical tables.
  - Oracle: per-turn `exe_ans`, numeric-tolerance; integers exact.
- `restate_in_scale(report, items) -> Record<item, number>` (TAT-QA `scale`): convert to units with an exact
  scale rule. Oracle: gold × scale.
- `verify_derivations(report, claims) -> boolean[]`: a claim is the FinQA `program` run on a perturbed operand
  vs the real one. Oracle: executed program results.

**26–27. HybridQA / WTQ / SQA / WikiSQL / TabFact**
- `table_and_links_brief(table, pages, questions) -> string[]`: lookup rows with crisp code, then per-linked-page
  `read_page` → `N<string>`. Several questions share one table world. Oracle: span-qa per question (golds of 8
  words or fewer).
- `fact_check_digest(table, statements[]) -> {statement; label}[]`: TabFact statements on one table. A per-statement
  `nl` plan picks columns/ops, executed in code. Oracle: entailed/refuted labels.
- SQA-style `drilldown(table, steps[])`: each step filters the previous answer set (carried state). Oracle:
  answer_coordinates sets per step.

**28. TableInstruct** (unused)
- `table_analyst(table, questions[]) -> answers` across `qtype` FactChecking / NumericalReasoning / DataAnalysis.
  Use `table` + `question` + `answer` only, never the PoT `instruction` prose. Oracle: normalized number / answer-list.
  Skip free-form DataAnalysis answers (correlation descriptions) unless they are numeric.
- `trend_label(table, column) -> "increase"|"decrease"|"no clear trend"` where the qsubtype is trend. Oracle: exact
  label, only where computable by a stated monotonic rule from visible values.

**29. Qasper** (unused; train only)
- `paper_qa(paper: Folder<section>, questions[]) -> (boolean|string|"unanswerable")[]`.
  - Per-section `skim` → `N<string>`; a `route` helper picks sections; answers are read from them.
  - Oracle: `yes_no` exact, extractive spans via span-qa (golds of 8 words or fewer), `unanswerable` exact.
    Free-form answers are skipped.
- `figure_lookup(paper, question)`, with captions from `figures_and_tables` as inputs: return the figure/table id
  that holds the evidence. Oracle: annotator `evidence` naming that caption.

**30. StrategyQA** (unused)
- `judge_with_facts(question, facts: string[]) -> boolean`: a `decompose` helper (the gold `decomposition` stays
  hidden; it is only a reference for the author), one `check_step` child per step reading only `facts`, then a
  crisp AND/OR combination. Oracle: the boolean `answer`. **Admit only rows whose `facts` suffice**: author review
  of each case; StrategyQA facts can be partial.

**31. MuSiQue / 2Wiki** (beyond single-call)
- `answerability_triage(questions[], paragraphs) -> {q; answer | "unanswerable"}[]`: MuSiQue `answerable=false`
  items are exact negatives. Per-paragraph notes are fanned in; a decision readout per question. Oracle: span-qa or
  exact "unanswerable".
- 2Wiki `evidences` triples: `build_mini_kg(paragraphs) -> triple[]`, then `query_kg`. Oracle: the gold triples
  (answer-list over normalized (s, r, o)) plus the final answer.

**32. QuALITY / TriviaQA / DROP**
- QuALITY `hard` items: `part_notes` per chunk (iterate_neuralese), then `eliminate_options` with per-option
  evidence. Oracle: the MCQ letter.
- DROP: `count_and_compare(passage, questions[])`: events extracted by `nl` into a typed list, then crisp
  counting/date arithmetic. Oracle: DROP number/date/spans.

**33. bAbI / CLUTRR / LogiQA**
- `track_world(story_lines, queries[]) -> answers[]`: iterateOn over lines with a carried typed world state (who
  holds what, where). Oracle: bAbI answers per query (tasks 2/3/7/8 have the deepest state).
- CLUTRR: `kinship_chain(story) -> relation`: an `extract_edges` `nl` helper, then crisp composition over a given
  composition table. Oracle: `target_text`. Admit only where the edge path is in the story (`proof_state`).

**34. Enigmata** (unused; port one verifier per family first)
- zebra_logic / campsite / star_battle: `solve_grid(clues) -> grid` with a `deduce` helper per clue carrying a
  candidate set (iterate_neuralese); crisp propagation. Oracle: the family's ported verifier, or the unique `answer`
  for unique-solution families.
- crypto_KPA/KKA: `decrypt(ciphertext, known_pairs) -> plaintext`. Oracle: exact plaintext.

**35. RLVR-IF constraints** (unused)
- `draft_notice(facts, constraints) -> string`: write a notice whose content facts come from inputs and whose form
  obeys the constraints (word count, keywords, sections). A `check_constraints` helper revises it until it passes
  (iterateOn with a measure). Oracle: the existing `constraints` checks **plus** exact presence of the required
  input facts. Keyword/length checks alone do not prove fidelity (AGENTS.md).

**36. WebInstruct / Numina / DeepScaleR / ORZ** (low natlang fit; use sparingly)
- `tutor_check(problem, student_steps[]) -> first_wrong_step | "correct"`: plant an arithmetic error into a worked
  solution. Oracle: the planted index (exact); the problem's answer stays visible as context.
- `batch_grade(answers: {problem; claimed}[]) -> boolean[]` with `answer_type` float/integer only. Oracle: exact
  numeric equivalence to gold. Decontaminate against MATH-500/AIME/AMC first.

## Where to start

The highest value is in the unused, agentic, state-carrying rows: tau-bench stores (1), APIGen-MT (2), tau2 telecom
train (3), the research dossiers (12–15), session reconstruction (9–11, 19), MAST postmortems (17), Spider/BIRD
multi-request (21). On the KR side: Qasper (29), StrategyQA with facts (30), TableInstruct (28), and multi-question
or multi-turn worlds over the financial and table sets (23–27). Rotate constructs per batch and author several
different operations per world, as the owner's codebase-unit rule asks.
