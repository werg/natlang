---
kind: directory-reducer
args:
  request: EditRequest
returns: EditResult
---
Make the edit that request.hypothesis describes in the program source held in folder, and report it. The goal describes what the edited program must do when it runs.

1. Read request.sourceFiles; they hold the current text of every existing source path. Read request.hypothesis for the kind of edit, the files to change and the expected effect, and request.diagnosis for the cases behind it.
2. Edit the paths in request.hypothesis.files, which are among request.allowedFiles. Create a path that does not exist yet by writing it. The files the program writes when it runs (such as report.json) belong to its runtime folder; express how the program produces them in its source.
3. Reorganize the execution structure as the hypothesis requires. The .nl interpreter judges visible inputs directly, so a root reducer can read a small batch, judge its meaning and do the exact bookkeeping in one final eval. Keep exact code for counting, parsing and arithmetic. Hand a semantic batch to a separate function when that subproblem stands on its own. When the hypothesis names a helper, read the helper and change or replace it.
4. Write each judgment as an instruction that reads the input's meaning. Take service field names and enum meanings from request.serviceDeclarations. Describe paths and counts as runtime data, so the program holds across the training folders.
5. When request.transformation is present, it names the kind of change this run performs; keep the edit within that kind and keep every behavior listed in request.preserves.
6. Return summary, one sentence naming the observed failure or wasted action the edit addresses, and preserves, the external behaviors the edit keeps.
