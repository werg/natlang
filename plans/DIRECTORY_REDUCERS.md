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

1. **Use the surface models know.** Paths, an editor, a shell, TypeScript and Python: the shapes of the largest agentic
   corpora. Our constructs (`nl`, `iterateOn`, reducers) are added to those languages rather than beside them, and the
   same loop and recursion policy holds in each.
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

The tools and the folder structure are open to change: nothing below keeps a current shape just because it exists.

A. **Delegation per file and per subfolder.** Make `nl` over a `FileHandle` or a `Folder` first-class. The child call
   gets that handle (read-only unless the parent's folder is writable) and the file tools scoped to it. Then
   `await Promise.all((await folder.files('inbox/*.eml')).map(file => nl<Label>`Classify the email in file.`(file)))`
   is the natural map, and `folder.dir(d).apply(reducer)` the natural map over subfolders. Put one example of each in
   the prompt, and have the generated references use them instead of reading everything into one eval.

B. **Document views.** Reading a `.pdf`, `.docx`, `.xlsx`, `.pptx` or `.html` gives a text rendering (read-only, with
   a note saying so); writes go to text formats. Conversion happens in the host (for example MarkItDown or pdftotext).
   Realistic workspaces (Workspace-Bench has 74 file types) need this; synthetic ones do not.

C. **A shell: bash on [just-bash](https://github.com/vercel-labs/just-bash/tree/main/packages/just-bash), forked.**

   **What it is.** just-bash (Apache-2.0) is a bash interpreter written in TypeScript. It runs in-process against a
   pluggable virtual filesystem (the `IFileSystem` interface, about 15 async methods, with lazy file providers) and has
   more than 80 commands built in: coreutils, grep, rg, sed, awk, sort, uniq, jq, yq, xan (CSV), diff and tar. It also
   has sqlite3 (sql.js) and python3 (CPython 3.13 for WebAssembly, in a worker).

   **Measured on 2026-09-27 (v3.4.2):**
   - text pipelines take 10–25 ms per command, sqlite3 about 70 ms and python3 about 200 ms;
   - writes land in the virtual filesystem, and sqlite3 works on database files inside it;
   - unknown binaries fail with "command not found", and `while true` stops at the command-count limit;
   - its parser (`parse`) gives a syntax tree with `While`, `Until`, `CStyleFor`, `For` and `FunctionDef` nodes, and
     transform plugins can check or rewrite a script before it runs;
   - `defineCommand` adds commands that take arguments and stdin, and they work inside pipelines.

   **Policy, as in eval.**
   - A script is parsed and checked before it runs. `while`, `until` and C-style `for ((…))` are refused with a
     message pointing to `for x in …` over a finite list and to `iterateOn` in eval or Python.
   - A function that calls itself, directly or through another, is refused.
   - `for f in *.md` and other `for … in` loops over words stay allowed. The command-count and time limits remain as a
     backstop.

   **The tool.**
   - `bash(command)` returns the exit code, stdout and stderr (capped and paged like other output) and the paths it
     changed.
   - The clock and file times are pinned, so a replay reproduces a command's output.
   - The fork cuts network (`curl`), `js-exec` (eval runs TypeScript), compression and HTML conversion. It replaces
     python3 with the Python of D and adds the `natlang` command of E.

   **What it unlocks:**
   - the command-line text processing models are strongest at;
   - direct conversion of terminal trajectories (Terminus-style command batches become `bash` calls, re-executed for
     their observations);
   - the bash half of the OpenHands SWE trajectories.

   **Costs:**
   - It runs in-process without VM isolation. just-bash has guards of its own (limits, prototype-pollution defences,
     no network), and both the commands and the hosts are ours.
   - In the browser the core shell runs; D's Python runs there too.

   **Alternatives checked.**
   - [@cloudflare/shell](https://github.com/cloudflare/agents/tree/main/packages/shell) (MIT, experimental) runs
     JavaScript in an isolated Worker over a state backend. It does not parse shell syntax, and eval already covers it.
   - [cloudflare/computer](https://github.com/cloudflare/computer) (MIT, preview) is a durable workspace filesystem
     with interchangeable execution backends. Its lightweight "isolate shell" backend is just-bash, which confirms the
     pattern rather than offering another engine.

D. **Python, with libraries and with natlang: [Pyodide](https://pyodide.org).**

   **What it is.** Pyodide (MPL-2.0) is CPython 3.14 for WebAssembly, with numpy, pandas and several hundred other
   packages as prebuilt wheels. It runs in Node and in the browser. Measured on 2026-09-27 (0.314):
   - it loads in 1.8 s; numpy and pandas add 2.2 s; the first pandas run takes 2.8 s, and later ones are fast;
   - `sqlite3` and `ast` work;
   - a JavaScript module registered as `natlang` is importable, and Python can `await` its async functions.

   So one warm instance per session is the right shape, with the packages vendored and locked (not fetched from a
   CDN at run time).

   **Python extended as TypeScript is.** The `natlang` module gives the same constructs:
   - `urgent = await nl[bool]("Is ticket urgent?")(ticket)`; with no type argument, the annotation of the target or
     later use types it, as in TypeScript, or the result is open;
   - `labels = await gather(*(nl[Label]("Label item.")(item) for item in items))`;
   - `final = await iterate_on(step, initial).until(lambda state: state.done)`, and `step.iterate_on(initial)`;
   - `await folder.dir("packages/api").apply(bump_version)` and `folder.files("inbox/*.eml")`, as in eval;
   - the call's arguments, functions and services are bound by name, as in eval.

   Instructions see the variables they name, as in TypeScript. The type inference and capture analysis are done over
   Python's own `ast`.

   **Policy, as in eval.**
   - `while` is refused, and so is a function that calls itself, directly or through another.
   - Every `for` and comprehension iterates a finite iterable: the lowering wraps its iterable, as `__natlang_finite`
     does in TypeScript, so `itertools.count()` and other endless iterators are refused at run time.

   **Where it runs.** One engine serves two places:
   - `python3` inside the shell, for scripts and one-liners over the folder's files;
   - Python as a second eval language, with the call's scope. This is either `eval` with a `language` argument or a
     `python` tool, whichever the corpora we convert favour; that is a decision.

   Python's file system is the call's folder (see F), so `open("notes/a.md")` and `pd.read_csv("sales.csv")` read the
   same files as the other tools.

E. **The `natlang` command, identical inside the shell and on the command line.**

   **Why not `nl`.** In the shell, `nl` is already coreutils' line numbering. So the command is `natlang`, and the real
   CLI takes the same subcommands and flags, reading stdin whenever it is not a terminal.

   **Subcommands:**
   - `natlang ask INSTRUCTION [--returns TYPE]` makes one call on stdin as a whole (the input), or on the folder when
     stdin is empty. This is today's `ask`, extended.
   - `--lines` makes one call per line and prints one result per line, in order. `--jsonl` does the same per JSON
     record, typed. `--files GLOB…` does it per file, printing `path<TAB>result`.
   - `--filter` prints the inputs whose judgment is true: grep by meaning.
   - `--jobs N` sets concurrency.
   - `natlang call NAME|FILE.nl [--lines|--jsonl]` calls one of the program's functions, or a `.nl` file, with inputs
     from stdin JSON or per line.
   - `natlang apply REDUCER DIR` runs a directory reducer on a subfolder.

   **Output.** Values print as plain text for strings, numbers and booleans, and as JSON otherwise, so pipelines
   compose.

   **Example:**
   `cat inbox/*.txt | natlang ask --lines --filter "asks for a refund" | wc -l`
   `natlang ask --files 'contracts/*.md' --returns 'yes|no' "Does the contract have a non-compete clause?" > nc.tsv`

   **Implementation.** A call from the shell is a child call like one from eval, recorded in the trajectory, and the
   CLI shares the implementation. There is no loop construct in the shell for open-ended repetition; that stays with
   `iterateOn`.

F. **The folder, redesigned around mounts and overlays.** Today's `Folder` (an overlay over a source, with change
   sets, transactions, handles and `apply`) grew for small reducers.

   **Requirements:**
   - lazy files, so a 100k-email corpus built from data is not materialized;
   - mounts, so a read-only corpus sits beside a writable workspace and a sub-reducer's folder is a view;
   - one change set across every tool;
   - adapters for just-bash (`IFileSystem`), Pyodide (an Emscripten filesystem mounted over it, instead of copying
     files in and out), Node disks and browser storage.

   **Options:**
   1. Keep `Folder` and write the three adapters over it.
   2. Rebuild `Folder` on a mount and overlay core modelled on just-bash's `InMemoryFs`, `OverlayFs` and
      `MountableFs` (lazy providers included), keeping our change sets, transactions and `apply` on top.
   3. Adopt just-bash's filesystem outright and move our features into the fork.

   **Recommendation:** start with option 1 (the adapters are small, and they measure the fit), then decide between 2
   and 3 with that experience. The API models see (`folder.file(…)`, `folder.dir(…).apply(…)`, the fs helper) stays,
   and gets a Python twin.

G. **The tool surface, aligned with the corpora.** The largest open agentic corpora use two shapes:
   - OpenHands: `str_replace_editor` (view, create, str_replace, insert) plus `execute_bash`;
   - Claude-Code style: Read, Write, Edit, Glob, Grep, Bash.

   Our file tools are close to the second. With a shell, listing and searching become `ls` and `rg`, which models use
   by reflex. So the proposal is to collapse the file tools to one editor tool in whichever shape the converted corpora
   use most (view with line numbers and ranges, create, exact replace, insert) plus `bash`. `eval` stays, Python per D
   is added, and `return_result`, `read_page` and `compact_history` stay. `diff_files` becomes a shell command
   (`changes`) that shows this call's changes. Converting trajectories then needs no renaming.

H. **Deliberately not added:** network access, git, and package installation at run time (the Python packages are a
   fixed, vendored set).

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
   and `Folder`; prompt examples; references switched to per-file delegation.
2. **The shell.**
   - A fork of just-bash with the policy checks, the `FolderFs` adapter and the `bash` tool.
   - The `natlang` command, built together with stdin support in the real CLI, so the two stay identical.
   - Pinned clock and replay.
3. **Python.**
   - Pyodide with a vendored, locked package set, and the `natlang` module (`nl`, `iterate_on`, folder handles, scope
     bindings).
   - The `ast`-based policy and lowering.
   - `python3` in the shell, then Python as an eval language.
   - The Emscripten mount over the folder.
4. **The folder core and the tool surface.** Decide between F's options with the adapters' experience, then collapse
   the file tools per G. Migrate recorded trajectories to the new tool names, as earlier surface changes were.
5. **Data.** The generated folder families and first-wave datasets (section 4), using the shell and Python where the
   task calls for them; the normalized and span oracle levels; teacher collection on dataset-backed folders.
   Conversion of the OpenHands and terminal trajectories (section 5).
6. **Document views and the evals** (Workspace-Bench-Lite, MuDABench).

Phases 1 to 3 can overlap: they touch different code.

## Decisions for you

1. **Python's place.** Should it be a second eval language (`eval` with `language: "python"`) or a separate `python`
   tool? The corpora favour a separate tool; one tool is more minimal.
2. **The tool collapse (G).** Should the file tools become one editor tool plus `bash`, and in which shape, OpenHands
   `str_replace_editor` or Claude-Code style?
3. **Softer oracles in training.** May normalized and span-level rows train, or only exact ones, with the softer
   levels kept for evaluation?
4. **Share of code-centric SWE trajectories.** They are the largest supply of file-editing data, but not semantic
   processing. Should they be included, and how much?
