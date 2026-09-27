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
5. **Browser parity.** Everything a call can do in Node it can do in the browser: shell, Python, sqlite, `natlang`.
   The one difference is where folders come from: in the browser they are in memory (data, archives, files the user
   hands over), never the real filesystem.
6. **Network as eval has it.** Network follows the runtime's `network` setting, on by default, in every tool.

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
   - The fork cuts `js-exec` (eval runs TypeScript), compression and HTML conversion. `curl` stays and follows the
     runtime's network setting.
   - It replaces `python3` with D's Python, and `sqlite3` with a small shim over Python's `sqlite3` module on the same
     interpreter, so one WebAssembly runtime serves both and both work in the browser. (just-bash's own `python3` and
     `sqlite3` are Node-only.) Python's built-in `python -m sqlite3` is not a drop-in: in the spike it refused a script
     ("You can only execute one statement at a time"). So the shim runs scripts with `executescript` and prints in the
     sqlite3 tool's formats: pipe-separated by default, plus `-header`, `-csv`, `-json` and `-separator`. sqlite in the
     browser can come after Python.
   - It adds the `natlang` command of E.

   **What it unlocks:**
   - the command-line text processing models are strongest at;
   - direct conversion of terminal trajectories (Terminus-style command batches become `bash` calls, re-executed for
     their observations);
   - the bash half of the OpenHands SWE trajectories.

   **Costs:**
   - It runs in-process without VM isolation. just-bash has guards of its own (limits, prototype-pollution defences),
     and both the commands and the hosts are ours.
   - In the browser, just-bash's core shell (parser, interpreter, built-in commands) runs as it is. With D's Python and
     the sqlite built on it, the fork has the same commands in the browser as in Node.

   **Alternatives checked.**
   - [@cloudflare/shell](https://github.com/cloudflare/agents/tree/main/packages/shell) (MIT, experimental) runs
     JavaScript in an isolated Worker over a state backend. It does not parse shell syntax, and eval already covers it.
   - [cloudflare/computer](https://github.com/cloudflare/computer) (MIT, preview) is a durable workspace filesystem
     with interchangeable execution backends. Its lightweight "isolate shell" backend is just-bash, which confirms the
     pattern rather than offering another engine.

D. **Python, with libraries and with natlang: [Pyodide](https://pyodide.org).** Spiked on 2026-09-27; the prototype
   and its tests are in `plans/spikes/python-folder/` until phase 3 replaces them.

   **Engine.** Pyodide (MPL-2.0) is CPython 3.14 for WebAssembly, with numpy, pandas and several hundred other packages
   as prebuilt wheels, and `sqlite3` in the standard library. It runs in Node and in the browser.
   - It loads in 1.8 s, numpy and pandas add 2.2 s, and a first pandas run takes about 2.8 s. So there is one warm
     instance per process, loaded on first use, with packages loaded from a cell's imports
     (`loadPackagesFromImports`).
   - Packages are vendored and locked, not fetched from a CDN at run time.

   **Where it runs.** On the host's main thread, not in a worker, so that file access stays synchronous and in-process
   (a worker would need a cross-thread call for every file operation). A small watchdog worker enforces time limits:
   it sets Pyodide's interrupt buffer, a shared array the interpreter polls. That stopped a pure-Python loop of 10¹⁰
   iterations after 0.55 s with `KeyboardInterrupt`, which the tool reports as a timeout.

   **Folder as filesystem.** An Emscripten filesystem is mounted over the call's folder, and Python works in it as its
   working directory. `open`, `pathlib`, `os.walk`, `shutil`, pandas and sqlite then act on the folder itself, and
   their changes appear in `diff`, transactions and `apply` like any other tool's.
   - **Built on MEMFS.** Its file nodes are MEMFS nodes, so reading, writing, seeking and mmap are MEMFS's code. Ours
     adds lookup and listing from the folder, contents loaded on first open, and write-back on close or sync.
     Directories are implicit, as in our folders: an empty `mkdir` lives in the mount until a file is written under it.
     File times are pinned.
   - **Talks to a minimal view,** which the folder core (F) provides: `kind(path)`, `entries(dir)`, `size(path)`,
     `read`, `write`, `remove`, `move`, and `revision()`, a counter that moves on every change.
   - **Coherence.** Another tool, or a child call, can change the folder only while Python is waiting on the host.
     Cached nodes are therefore dropped (from Emscripten's name table too) at exactly those points: when Python is
     entered, and when it resumes after awaiting a host call. An open file keeps its contents, as an open descriptor
     does. A later open reloads it if the folder changed and it holds no unsaved writes.
   - **Measured over a 10k-file folder:**

     | Operation | Time | File contents read |
     |---|---|---|
     | list a directory | 14 ms | none |
     | walk the folder | 34 ms | none |
     | stat all 10k files | 83 ms | none (sizes from the source) |
     | read 1k files | 40 ms | 1k |

     Also checked: `open`/append, `rglob`, `makedirs`, pandas `to_csv`, rename, delete, `shutil.copy`, sqlite with its
     journal, `FileNotFoundError`, and a change made outside becoming visible on the same mount.
   - **One fix is required:** MEMFS can hold a view of WebAssembly memory (an `Int8Array` over the heap) that later
     writes reuse, so write-back copies contents out as a fresh `Uint8Array`.
   - **To add:** evicting clean cached contents past a memory budget, since a 100k-file corpus read end to end would
     otherwise stay in the WebAssembly heap.

   **Concurrent calls.** Parallel calls (say, a per-file `nl` map whose children each use Python) share one
   interpreter, which is single-threaded but interleaves at `await`s.
   - Each call gets its own mount (`/calls/<id>`) and its own namespace.
   - The working directory is global to the interpreter. So the same resume hook that refreshes the mount also restores
     the resuming call's working directory, and the local-module import hook resolves against it.

   **In the browser.** The same design, with three differences:
   - Python runs on the runtime's thread, which in the browser is the natlang worker (I), so it never blocks the page.
   - The watchdog's interrupt needs `SharedArrayBuffer`, which browsers grant only to cross-origin-isolated pages
     (COOP/COEP, which multithreaded inference already asks for). Without isolation, the loop caps still bound every
     loop, but a time limit cannot stop a running computation.
   - `wait()` needs JavaScript Promise Integration: shipped in Chrome, not yet in every browser. Where it is missing,
     `wait()` raises and says to `await` instead. Async code works everywhere.

   Pyodide and its packages are served beside `natlang.js` and loaded on first use (numpy and pandas are tens of MB).
   Folders are in memory; Pyodide's native-filesystem mount (`NATIVEFS`) is not used.

   **The `natlang` module.** It gives the host's own objects Python's shapes; it adds no second implementation. Checked
   in the spike:
   - `await nl[bool]("Is ticket urgent?")(ticket)`. `nl[T]` takes Python types: `bool`, numbers, `str`, `Literal`,
     unions, `list`, `dict`, `TypedDict` and dataclasses become natlang type text, for example
     `list[Ticket]` → `{ id: string, text: string }[]`. Plain `nl(...)` is an open result, as in TypeScript.
   - `wait(nl[bool]("Is it spam?")(m))` runs a call from synchronous code, such as a function pandas applies. It uses
     Pyodide's `run_sync`, which needs JavaScript Promise Integration (`--experimental-wasm-jspi` on Node 24.0; on by
     default in current browsers). It worked inside `DataFrame.apply`.
   - `await iterate_on(step, initial).until(done)` is the host's `iterateOn`, with its progress reviews, limits and
     trace. The Python step and check are kept alive (`create_proxy`) for the whole run and released after it: a
     plain Python callable passed to JavaScript is destroyed when the call returns.
   - Folder handles are plain `pathlib.Path`s into the mount, so the standard library is the folder API.
     `folder("packages/api").apply(bump_version)` and `nl` over a path hand a child call that subfolder or file,
     rebased as in A.
   - The call's arguments, functions and services are bound in the namespace by name, as in eval.
   - Later: static inference of `nl`'s type from the target's annotation (`level: Literal[...] = await nl(...)(x)`),
     over `ast`, as the TypeScript compiler does from contextual types.

   **Policy, as in eval.** Checked in the spike; the check and lowering are about 100 lines over `ast`.
   - Refused, with messages that say what to use instead: `while`; a function that calls itself, directly or through
     another in the module; calls to `exec`, `eval`, `compile` and `breakpoint`; imports of `js`, `pyodide_js`,
     `pyodide.ffi`, `micropip`, `importlib` and `ctypes`.
   - Every `for` loop and comprehension goes through `__natlang_finite`. Concrete collections pass as they are: list,
     tuple, str, bytes, dict and its views, set, frozenset, range, and numpy and pandas objects. Anything else (a
     generator, `zip`, `itertools` objects) is taken up to 100k items, and beyond that raises with a pointer to
     `iterate_on`. The spike's `itertools.count()` loop stopped at the cap; `zip`, generators and `islice` passed.
   - **Scope: user code only.** Cells and local modules run with builtins that lack `exec`, `eval` and `compile`, so
     looking one up by name fails. Library code keeps the real builtins, which it uses (`dataclasses` runs `exec`),
     and Pyodide's own modules stay importable for the libraries that need them. Removing them globally broke both in
     the spike.
   - Local modules imported from the folder are checked and lowered by an import hook (a meta-path finder). A module
     with a `while` loop was refused on import.
   - **Entry points:** the `python` tool's cells, and `python3` in the shell (scripts in the folder and `-c`), both on
     the same interpreter.
   - **Network follows the runtime's setting,** as `fetch` in eval does: on by default. When it is on, `pyfetch` and
     patched `requests` work; when a host turns it off, the `fetch` they use is absent. Pyodide is loaded with a
     `jsglobals` object that carries only what the host grants, so even Pyodide's own foreign-function interface
     reaches no `process`, `require` or host objects.

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
   - adapters for just-bash (`IFileSystem`) and Pyodide (the Emscripten filesystem of D, over the same view);
   - sources: Node disks, and in memory (data, archives, files handed over) in both Node and the browser, where
     nothing reaches the real filesystem;
   - listing and stat that do not read contents. Today `Folder.statSync` hashes every file to fill in `digest`, and
     `list()` calls it for every entry, so listing a folder reads all of it. Sizes come from the source instead, and
     digests are computed when a diff needs them. The spike's view (D) shows the shape: a path index, sizes from the
     source, and a revision counter.

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

I. **In the browser, natlang runs in a worker.** The page's main thread holds only a thin client and the DOM renderer;
   the runtime runs in a dedicated worker by default, not in the page. Python, the shell and long calls then never
   block the page, and D's synchronous file access and watchdog work unchanged, since the worker is the runtime's
   thread.
   - **What already fits.** Eval runs through `new Function`, which workers have. Natural-language applications
     already reach the DOM only through data: they produce `UiNode` trees, and `BrowserDomRenderer` draws them and
     dispatches actions back. The event loop moves to the worker, the renderer stays on the page, and trees and actions
     cross as messages. Local inference (wllama) is already a worker; the runtime worker starts it as a nested worker,
     and WebGPU and the model cache (OPFS) are available to workers.
   - **The client API.** `createNatlangWorker(options)` returns a client with the shape of today's module (load a
     model, compile a project, call functions, run an application's event loop, make folders), implemented as messages
     to the worker.
     - Values cross by structured clone. File contents and folder data cross as transferred buffers, and a folder's
       changes come back as a change set.
     - Streams (steps, traces, iteration events) come back over a `MessagePort`.
     - Cancelling a call aborts it in the worker, and interrupts Python when the page is cross-origin isolated.
   - **Host services.** Services and functions an application defines on the page reach the worker as asynchronous
     proxies. Eval already awaits what it is handed, but a service that must answer synchronously belongs in the worker.
     The guide says so, and the scaffolds put services in the worker.
   - **Data** that is large (a corpus, an archive) is fetched or unpacked in the worker, not copied across.
   - **Cross-origin isolation** (COOP/COEP) is recommended: it enables `SharedArrayBuffer`, and with it Python's
     interrupt and multithreaded inference. Without it, everything still runs, with the fallbacks in D.
   - **Running on the page itself** stays possible for tests and the smallest demos, but it is not the default in the
     documentation, the examples or `natlang` app scaffolds.

J. **Deliberately not added:** git, and package installation at run time (the Python packages are a
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

The browser worker (I) comes before Python reaches the browser: phase 3's browser work runs on it. It is also useful
on its own, for inference-heavy pages, so it can start any time.

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
