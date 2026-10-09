---
args:
  request: HypothesizeRequest
returns: Hypothesis
---
Choose the one edit to the program source that the diagnosis supports, and describe it. A hypothesis is a testable explanation with the edit that follows from it.

1. Read request.diagnosis.pattern, then the text of the files in request.sourceFiles that the observations point to.
2. Choose the smallest edit that explains the pattern. `instruction` rewrites an instruction. `helper` corrects exact code such as a join, a file pattern, a superseded record or a trimmed edit string. `structure` reorganizes the program, for example a root reducer that reads a small batch, judges its meaning directly and does the exact bookkeeping in one final eval; use `structure` when request.mode is `structural`. `efficiency` removes a model request or tool call that the results show to be unnecessary.
3. For an observation of category `execution-method`, choose the edit that makes the program use its ordinary file tools or scoped folder handles.
4. Compare with request.history. Choose an edit that differs from every rejected outcome in kind or in files.
5. Set files to the paths the edit changes. Every path is one of request.allowedFiles.
6. Set statement to one sentence naming the observed failure or wasted action the edit addresses. Set predictedChange to the measurement you expect to move.
7. When request.transformation is present, it names the kind of change this run performs; choose the edit within that kind.
8. When the evidence supports no edit, return kind `none`, an empty files list and a statement that says why.
