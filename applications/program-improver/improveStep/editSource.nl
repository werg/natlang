---
kind: directory-reducer
args:
  request: EditRequest
returns: EditResult
---
Make the edit that request.hypothesis describes in the program source held in folder, and report it. The goal describes what the edited program must do when it runs.

1. Use request.sourceFiles directly; they hold the full current text of every existing source path. Start from request.brief and open a clipped part or an unresolved failure only where the hypothesis needs it. Read request.hypothesis for the kind of edit, the files to change and the expected effect, and request.diagnosis for the cases behind it.
2. Edit the paths in request.hypothesis.files, which are among request.allowedFiles. A path in request.allowedFiles that does not exist yet is a new file; create it by writing it. Write each complete replacement once. The files the program writes when it runs (such as report.json) belong to its runtime folder; express how the program produces them in its source.
3. Change prose only: rewrite the instructions in the natural-language text, and keep TypeScript, callable contracts and interpolation bindings as they are. State each rule once, in its narrowest scope; merge instructions that repeat each other and resolve instructions that contradict each other.
4. Write each judgment as an instruction that reads the input's meaning. Take service field names and enum meanings from request.serviceDeclarations. Describe paths and counts as runtime data, so the program holds across the training folders.
5. When request.transformation is present, it names the kind of change this run performs; keep the edit within that kind and keep every behavior listed in request.preserves.
6. When the evidence supports a coherent edit, write it and finish.
7. Return summary, one sentence naming the observed failure or wasted action the edit addresses, and preserves, the external behaviors the edit keeps.
