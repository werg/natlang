# Directory reducers: data, affordances and training

Status: plan, 2026-09-27. Nothing here is implemented yet.

## Why

Code is in distribution for language models, and so is an agent editing files in a folder while running code there.
A directory reducer puts semantic work in that setting: the model reads files, judges, extracts, edits and writes
conclusions, with code for the exact parts. The aim is to get further with small models on semantic computation by
asking for work they have seen a great deal of. That covers real directories, and also any larger dataset that can be
laid out as a folder tree: an inbox, a contract archive, a set of meeting notes, a table split into records.

## Where we are

- **Surface.** A reducer (`kind: directory-reducer`) gets a private copy of its folder, the file tools (`list_files`,
  `search_files`, `read_file`, `write_file`, `edit_file`, `diff_files`) and, in eval, `folder` handles plus the `fs`
  helper. `folder.dir(p).apply(reducer)` runs a sub-reducer on a subfolder, and the changes present when it replies
  done are kept (`scoped-fs.ts`, `DIRECTORY_REDUCER_PROMPT`).
- **Host.** `openFolder(path)` gives a lazy disk folder, `saveFolder(path, changes)` writes changes back atomically,
  and `Folder.fromFiles({...})` builds one in memory. Nothing builds a folder from data.
- **Delegation.** The inline compiler already treats `Folder` and `FileHandle` as handle types, so an `nl` can in
  principle take a file or a folder. Neither the prompt nor any training row shows this.
- **Training data.** Folder work is a sliver: `folder_criteria_reducer`, `reducer_apply`, `event_retry`-style
  failures and the five authoring families, each with about five small files. The references read every file in one
  eval and decide directly, which is the opposite of what larger folders need.

## Principles

1. **Reuse the file surface models know.** Paths, list, search, read ranges, exact-span edits and diffs. New tools
   only where a whole class of in-distribution work is out of reach without them.
2. **Folders bigger than the context.** Tasks should be solvable only by navigating (listing, searching, reading
   parts) and by delegating per file or per subfolder. A 16k student cannot read a 200-file inbox.
3. **Exact oracles first.** A row is admitted by a check, as now. Where a check has to be softer (normalized
   answers, span overlap, a judge), it is a separate evidence level, recorded on the row.
4. **One way in.** A host brings a folder in with one call, whether it is a directory, an archive or data.

## 1. Bringing data in

- `openFolder(path)`, which exists; `openArchive(zip|tar)`; and `Folder.fromData(value, layout)` for everything else.
- **Layouts** are small declarative maps from records to paths and file bodies:
  - one file per record, e.g. `emails/{folder}/{id}.eml` or `contracts/{party}.md`
  - records grouped into directories by a field
  - a table as one CSV plus optional per-row files
  - front matter (YAML) for typed fields above free text
- **The reverse, `folderToData(folder, layout)`,** reads results back: moved files become labels, edited front matter
  becomes updated records, written reports become outputs. Callers never parse file trees themselves.
- **The same layouts serve three uses:** the curriculum generators synthesize folders with them, eval sets are built
  with them from datasets, and host applications use them. There is one implementation, with tests for round trips.

## 2. Affordances

In order of value per cost.

A. **Delegation per file and per subfolder.** Make `nl` over a `FileHandle` or a `Folder` first-class. The child call
   gets that handle (read-only unless the parent's folder is writable) and the file tools scoped to it. Then
   `await Promise.all((await folder.files('inbox/*.eml')).map(file => nl<Label>`Classify the email in file.`(file)))`
   is the natural map, and `folder.dir(d).apply(reducer)` the natural map over subfolders. Put one example of each in
   the prompt, and have the generated references use them instead of reading everything into one eval.

B. **Document views.** Reading a `.pdf`, `.docx`, `.xlsx`, `.pptx` or `.html` gives a text rendering (read-only, with
   a note saying so); writes go to text formats. Conversion happens in the host (for example MarkItDown or pdftotext).
   Realistic workspaces (Workspace-Bench has 74 file types) need this; synthetic ones do not.

C. **A `run` tool: a sandboxed shell, with Python through it.** This is the one real expansion. It would:
   - run one command in a checkout of the folder, with the changes read back as a change set
   - offer coreutils, grep, sed, awk, jq, sqlite3 and python3 with the common data libraries
   - have no network, a time limit and an output cap
   - replay deterministically from its journaled output, as model turns already do

   It would unlock three things: the command-line text processing models are strongest at; data work in Python and
   SQL, where models are far stronger than in TypeScript; and conversion of the large terminal and SWE trajectory sets
   (section 5). The costs are sandboxing (bubblewrap, or a container per call), keeping the folder overlay and the
   checkout consistent, and a second code path beside eval that the prompt must explain without muddying "code runs
   in eval". This is a decision point for you (see the end).

D. **Deliberately not added:** network access, git, package installation inside reducers, and a second editing
   model such as patch files. `edit_file` with exact spans stays the one way to change text.

## 3. Oracles and admission

| Evidence level | Check | Used for |
|---|---|---|
| exact | result equals the expected value; files equal the expected files | generated folders, moves, labels, CoEdIT and CommitPack edits |
| normalized | answer matches the gold or an alternate after normalization (case, whitespace, number formats, dates) | EnronQA, HotpotQA, MuSiQue, table QA |
| span | extracted spans overlap the annotated ones (F1 at or above a threshold) | CUAD clauses, extraction reports |
| judged | a strong model grades against a rubric and the gold, and the verdict is recorded | summaries, reports, knowledge-base writing; eval only at first |

`admitRow` gets the evidence level from the case (`semantics.oracle`), and the materializer copies it onto every
turn, so a training build can choose which levels to include.

## 4. Data sources

Priorities: **A** first wave, **B** second wave, **E** evaluation only.

| Source | License | Size | As a folder | Tasks | Oracle | Priority |
|---|---|---|---|---|---|---|
| Our labeled sets (SMS Spam, BANKING77, CLINC150, SST-2, AG News, emotion) | as recorded | ~100k items | items as files, grouped arbitrarily | triage into folders, tag front matter, index files with counts, find and quote | exact | A |
| [CoEdIT](https://huggingface.co/datasets/grammarly/coedit) | Apache-2.0 | ~70k sentence edits (grammar, simplify, paraphrase, coherence, neutralize) | edits composed into multi-paragraph documents in several files | "fix the grammar in drafts/", "simplify the sections marked …" | exact (target text) | A |
| [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) | MIT (per-repo permissive) | 2 GB, 277 languages, including Markdown, YAML, JSON, reStructuredText and TeX | the file at its repo path, optionally with siblings | the commit message as the instruction; the reference is `edit_file` spans derived from the diff | exact (new file) | A |
| [CUAD](https://huggingface.co/datasets/theatticusproject/cuad) | CC-BY-4.0 | 510 contracts, 41 clause types | `contracts/*.md` | "which contracts have a non-compete; quote it", "write clauses.csv" | span or exact (yes/no) | A |
| HotpotQA (CC-BY-SA-4.0), MuSiQue (CC-BY-4.0) | as noted | ~90k and ~25k questions | the question's paragraphs plus distractors as `wiki/*.md` | multi-hop questions that need search and reading several files | normalized | A |
| [EnronQA](https://huggingface.co/datasets/MichaelR207/enron_qa_0922) | CC (variant to confirm) | 103k emails, 528k QA pairs, 150 inboxes | a maildir per user (the corpus is one) | QA over an inbox; the original Enron mailbox folders give real "file these emails" labels | normalized (gold plus alternates); exact for folder moves | A |
| WikiTableQuestions (CC-BY-SA-4.0), BIRD (CC-BY-SA-4.0), [InfiAgent-DABench](https://arxiv.org/pdf/2401.05507) | as noted (DABench to confirm) | ~22k, ~12k, 257 questions | CSV or SQLite files in a folder | questions mixing code over tables with judgment over text columns | normalized | B (much better with `run`) |
| [QMSum](https://github.com/Yale-LILY/QMSum) | MIT | 232 meetings, 1.8k queries | `meetings/*.md` transcripts | query-focused summaries, decision logs | judged | B |
| [Workspace-Bench](https://github.com/OpenDataBox/Workspace-Bench) | Apache-2.0 | 20k files, 5 roles, 388 tasks (Lite: 100) | real workspaces, 74 file types | under-specified workplace tasks across many files | rubric | E, plus workspaces to write tasks over |
| [MuDABench](https://github.com/Zhanli-Li/MuDABench) | Apache-2.0 | 80k pages, 332 questions | documents per entity | analytical QA across many documents | normalized | E |

## 5. Existing agentic SFT, bent to our surface

The trajectories are worth more than their tasks only where their actions translate.

- **OpenHands and SWE-agent SWE trajectories** ([SWE-rebench-openhands](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories),
  CC-BY-4.0; [SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories), 5k;
  SWE-Zero, 318k).
  - **Translation:** `str_replace_editor` calls map almost exactly onto our tools (view → `read_file`,
    create → `write_file`, str_replace → `edit_file`), and `grep` or `find` calls in bash map onto
    `search_files`/`list_files`.
  - **Heavy edit:** check out the repo at the task's commit as a folder, and replay the translated actions through our
    runtime to regenerate observations.
  - **Test runs:** drop them, or map them to `run` once it exists.
  - **Oracle:** the final files equal the trajectory's verified patch, so no tests need to run.
  - **Scope:** keep trajectories whose actions all translate. This is code, not semantic processing, but it is the
    largest supply of in-distribution file editing, and a first filter could keep only prose-heavy repos.
- **Terminal trajectories** ([OpenThoughts-Agent-SFT-100K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K),
  Apache-2.0, 94k; [LiteCoder-Terminal-SFT](https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-SFT),
  MIT, 11k).
  - These are batches of bash commands. They translate only with `run`, and their environments would need rebuilding.
  - Without `run`, mine the tasks and checkers, and have our teachers solve them in our environment.
- **Not usable:** [Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) (tool
  calling, search and customer service; no files) and similar general tool-calling sets.

## 6. Generated folder families

These follow the composed and labeled families: every part carries its code, its sentence and its layout, and the
references replay through the real runtime.

| Family | Folder | Task | Reference |
|---|---|---|---|
| `folder_triage` | 20–200 labeled items as files | move each into `by-<label>/`, or tag its front matter | per-file `nl` (labels from data); moves in code |
| `folder_index` | the same | write `INDEX.md` or `summary.csv` with counts and lists | search and `nl` judgments; the file written with code |
| `folder_edit` | CoEdIT documents across files | apply an edit instruction to the files it concerns | `read_file`, then exact `edit_file` spans |
| `folder_find` | HotpotQA or MuSiQue paragraphs with distractors | answer the question | `search_files`, targeted reads, answer |
| `folder_extract` | CUAD contracts | fill a CSV of clause answers | per-contract `nl` over the file handle |
| `folder_mixed` | CSV plus text files | e.g. total the amounts of the invoices whose notes dispute the charge | code over the CSV, `nl` over the notes |

Sizes are drawn so that about half the cases exceed what fits in context. As now, a child's judgment is a direct
answer, trained only with `--direct-answers`.

## 7. Evaluation

- **Held-out seeds of each generated family**, as now.
- **EnronQA and CUAD test splits**, with users and contracts disjoint from training.
- **Workspace-Bench-Lite and MuDABench**, never trained on, scored with their own rubrics.

## 8. Phases

1. **Folders from data, and delegation.** `Folder.fromData` and `folderToData` with layouts; `nl` over `FileHandle`
   and `Folder`; prompt examples; references switched to per-file delegation. No new tools. Test by round trips and by
   reference replay.
2. **First-wave families and datasets.** The generated folder families, plus CoEdIT, CommitPackFT (prose and config
   first), CUAD, HotpotQA/MuSiQue and EnronQA, with the normalized and span oracle levels. Teacher collection (Bonsai
   and Luna) on the dataset-backed folders where scripted references are impossible.
3. **Document views** (read-only text renderings) and the Workspace-Bench and MuDABench evals.
4. **`run`,** if chosen: sandbox, change-set sync, replay, prompt text; then table and data-analysis tasks, and
   conversion of the terminal and SWE trajectories.

## Decisions for you

1. **`run` (a sandboxed shell with Python).** It is the only large expansion, and it gates section 5's trajectory
   conversion and the table tasks. Should it be in scope, and is Python through `run` acceptable as the second
   language, or should Python get its own eval?
2. **Softer oracles in training.** May normalized and span-level rows train, or only exact ones, with the softer
   levels kept for evaluation?
3. **Share of code-centric SWE trajectories.** They are the largest supply of file-editing data, but not semantic
   processing. Should they be included, and how much?
