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
- **Identity and round trips.**
  - Every record has a stable id: the layout names a unique key, and the id is also written into the file's front
    matter, so a moved or renamed file keeps its identity.
  - A key that is not unique fails the build, unless the layout names a disambiguation (a deterministic suffix).
  - `folderToData` identifies records by their front-matter id (by path only for formats without front matter).
    A record whose file is gone is reported as deleted. A file that matches no record is returned as unknown,
    never silently dropped.
  - A layout has one writable representation: either the table or the per-record files. The other is a read-only
    view regenerated from it, so there is no precedence between two edited copies.
- **The same layouts serve three uses:** the curriculum generators synthesize folders with them, eval sets are built
  with them from datasets, and host applications use them. There is one implementation, with tests for round trips.

## 2. Affordances

The tools and the folder structure are open to change: nothing below keeps a current shape just because it exists.

A. **Delegation per file and per subfolder.** Make `nl` over a `FileHandle` or a `Folder` first-class. The child call
   gets that handle (read-only unless the parent's folder is writable) and the file tools scoped to it. Then
   `await Promise.all((await folder.files('inbox/*.eml')).map(file => nl<Label>`Classify the email in file.`(file)))`
   is the natural map, and `folder.dir(d).apply(reducer)` the natural map over subfolders. Put one example of each in
   the prompt, and have the generated references use them instead of reading everything into one eval.

   **Evidence.** A child that gets a handle must see what is in it before it answers.
   - A small file (up to a few thousand characters) is shown in the child's opening, as argument values are now.
   - A larger file or a folder is not; the child reads or searches it first.
   - Scripted references do the same: a per-file child's reference reads, then answers.
   - Admission checks it: a case names, per child, markers from the evidence (a quote from the file), and a child
     whose answer comes before any output showing them is refused. This extends today's `missing_observation` check,
     which covers only the root call, to children.

   **Capabilities and transactions** (settled in phase 1, before anything uses them).
   - A handle passed to a child is rebased: the child gets a folder rooted at the handle's path, with no `.folder`,
     `.parent` or path above its root. Today's handles keep their whole backing folder and expose `parent`, so passing
     one as it is does not scope anything.
   - A child given several handles gets each as its own root.
   - A read-only parent gives read-only handles.
   - A writable child works on its own overlay. When it finishes successfully, its change set is committed into the
     parent's overlay; when it fails, its changes are discarded.
   - Writes are locked per subtree, not per root. Today one writer lock covers the root and every copy made from it,
     so sibling subfolder reducers serialize. With subtree locks, `a/` and `b/` commit independently; a commit that
     touches a path another commit changed since its copy was taken fails with the existing `FolderConflictError`.

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
   - Every `for` loop and comprehension goes through a guard (the lowering wraps its iterable, as `__natlang_finite`
     does in TypeScript). Concrete collections pass as they are: list, tuple, str, bytes, dict and its views, set,
     frozenset, range, and numpy and pandas objects. Anything else (a generator, `zip`, `itertools` objects) is taken
     up to a cap of 100k items, and a longer one raises with a message pointing to `iterate_on`. The TypeScript guard is
     stricter, a whitelist; Python code iterates generators too routinely for that.

   **Entry points and capabilities.**
   - Code runs only through the `python` tool's cells and through `python3` in the shell (scripts in the folder and
     `-c`). Every source that runs passes the check and lowering, including a module imported from the folder: an
     import hook applies them to local sources.
   - `exec`, `eval`, `compile` and `__import__` of strings, and `importlib`, are refused.
   - So are `js`, `pyodide.ffi`, `pyodide.http`, `pyodide_js` and `micropip`. Pyodide is loaded with a restricted
     `jsglobals` object, so even the foreign-function interface reaches no `fetch` and no host object beyond the
     `natlang` module.
   - Only the vendored packages import.
   - Pyodide runs in a worker, so a time limit or a cancelled call interrupts it (Pyodide's interrupt buffer) without
     stopping the host.

   **Where it runs.** One engine serves two places:
   - `python3` inside the shell, for scripts and one-liners over the folder's files;
   - a separate `python` tool (decided), a persistent session with the call's scope, beside `eval`.

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

G. **The tool surface: one editor tool plus `bash`** (decided 2026-09-27). The file tools collapse:
   - listing and searching become `ls` and `rg`;
   - reading, creating and exact-replace editing become one editor, with view (line numbers, ranges), create,
     str_replace and insert;
   - `diff_files` becomes a shell command, `changes`, that shows this call's changes.

   `eval` stays, Python (D) is a separate `python` tool (decided), and `return_result`, `read_page` and
   `compact_history` stay.

   **The editor's shape is chosen by measurement.** The largest open corpora use OpenHands' `str_replace_editor`
   (one tool with a `command` argument); Claude-Code style splits it into Read, Write and Edit. The semantics are the
   same, and conversion between them is mechanical. So the editor is implemented once and rendered in both shapes. The
   untrained student and the teachers run a small folder probe with each, and the shape with fewer malformed calls and
   more solved tasks wins. Recorded trajectories are then migrated to it.

H. **A `delegate` tool: a directory-reducer subagent without code.**
   `delegate(path, instructions, returns?)` runs a directory reducer on a subfolder, with its own context and the same
   tools, and returns its result. Its changes are kept, as with `folder.dir(path).apply(reducer)`, which is what it
   lowers to. Models are post-trained with subagent tools of this shape (Claude Code's Task tool), and it is the way a
   long task over a large folder stays within a small model's context. The code form stays for maps over many
   subfolders; the tool is for the one-off delegation.

I. **Deliberately not added:** network access, git, and package installation at run time (the Python packages are a
   fixed, vendored set).

## 3. Oracles and admission

| Evidence level | Check | Used for |
|---|---|---|
| exact | result equals the expected value; files equal the expected files | generated folders, moves, labels, CoEdIT and CommitPack edits |
| normalized | answer matches the gold or an alternate after normalization (case, whitespace, number formats, dates) | EnronQA, HotpotQA, MuSiQue, table QA |
| span | extracted spans overlap the annotated ones (F1 at or above a threshold) | CUAD clauses, extraction reports |
| judged | a strong model grades against a rubric and the gold, and the verdict is recorded | summaries, reports, knowledge-base writing |

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
| WikiTableQuestions (CC-BY-SA-4.0), BIRD (CC-BY-SA-4.0), [InfiAgent-DABench](https://arxiv.org/pdf/2401.05507) | as noted (DABench to confirm) | ~22k, ~12k, 257 questions | CSV or SQLite files in a folder | questions mixing code over tables with judgment over text columns | normalized | B (with `bash` and `python`) |
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
  - **Actions that don't replay.** Test runs, package installs, and commands neither `bash` nor `python` can run
    (arbitrary test suites need the repo's dependencies, C extensions included, which Pyodide does not have) are not
    dropped from the middle of a trajectory: a later edit may depend on their output.
    - A trajectory whose every action translates and replays is kept whole.
    - Otherwise it is cut before the first action that does not replay, and the continuation is recollected by our
      teacher from that state in our runtime, with the final-patch oracle (the collector's hand-off does this now for a
      student's failed state).
    - Failing that, the trajectory is dropped.
  - **Oracle:** the final files equal the trajectory's verified patch, so no tests need to run.
  - **Scope:** keep trajectories whose actions all translate. This is code, not semantic processing, but it is the
    largest supply of in-distribution file editing, and a first filter could keep only prose-heavy repos.
- **Terminal trajectories** ([OpenThoughts-Agent-SFT-100K](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K),
  Apache-2.0, 94k; [LiteCoder-Terminal-SFT](https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-SFT),
  MIT, 11k).
  - These are batches of bash commands. They translate to `bash` where every command is one just-bash has (its 80-odd
    built-ins plus `python3` from Pyodide's package set); their environments are rebuilt from the task files.
  - Tasks that need other binaries (compilers, package managers, services) are not converted. Their tasks and
    checkers can still be mined for our teachers to solve in our environment.
- **Not usable:** [Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) (tool
  calling, search and customer service; no files) and similar general tool-calling sets.

### Long trajectories broken into delegated sub-tasks

A long trajectory is better training data for a small model as a short parent that delegates plus short children.

- **Mechanically, by subfolder.** Find maximal segments of a trajectory whose actions stay inside one subfolder, for
  example a run of reads and edits under `src/parser/`. Replace each segment in the parent with one `delegate` call on
  that subfolder; the segment becomes the child's trajectory. The child's instructions are written afterwards by a
  teacher from the segment's actions and diff, as a request that would lead to them.
- **Checked exactly.** Replay the child's instructions: the child must reproduce the segment's changes, and the parent
  must reach the original end state. Both are exact oracles, whatever the original task's oracle was.
- **Existing subagent trajectories.** Where a corpus already contains subagent calls (the Claude Code trajectories in
  LiteCoder-Terminal, for example), they map directly onto `delegate`.
- **The same applies to our own long teacher runs,** and to terminal trajectories once they are converted.

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

Sizes are drawn so that about half the cases exceed what fits in context. A per-file child reads its file (or sees a
small one in its opening) before it answers, per 2A. A child's answer given right after that evidence, without
reasoning towards it, is still a direct answer, trained only with `--direct-answers`.

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
   - `python3` in the shell, then the `python` tool.
   - The Emscripten mount over the folder.
4. **The folder core and the tool surface.** Decide between F's options with the adapters' experience, then collapse
   the file tools per G, including the editor-shape probe, and add `delegate` (H). Migrate recorded trajectories to the new tool names, as earlier surface changes were.
5. **Data.** The generated folder families and first-wave datasets (section 4), using the shell and Python where the
   task calls for them; the normalized and span oracle levels; teacher collection on dataset-backed folders.
   Conversion of the OpenHands and terminal trajectories (section 5), broken into delegated sub-tasks.
6. **Document views and the evals** (Workspace-Bench-Lite, MuDABench).

Phases 1 to 3 can overlap: they touch different code.

## Decisions

Taken on 2026-09-27:
- Bash loops (`while`, `until`, C-style `for`) and recursion are refused, as in eval; so are Python `while`, recursion
  and endless iteration.
- Python and bash both get natlang's constructs; the `natlang` command is identical in the shell and on the CLI.
- Python libraries are in scope (Pyodide).
- Python is a separate `python` tool.
- The file tools collapse to one editor plus `bash`; the editor's shape is decided by the probe in G.
- Training is not limited to exact oracles: normalized, span and judged rows may train, with their evidence level
  recorded so that builds can weight or filter them.
- Code-centric SWE trajectories are included, to train coding and file editing, preferably broken into delegated
  sub-tasks.
- The tools and the folder structure may be changed wholesale.
