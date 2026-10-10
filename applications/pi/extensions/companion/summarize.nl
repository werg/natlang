---
description: What a workspace file is for, its main definitions, and what someone changing or using it must know.
args:
  path: string
  text: string
returns: FileSummary
---
text is the content of the workspace file path (cut at a limit when the file is long; then it ends with a note saying
so). Return:
- purpose: what the file is for, in one or two sentences.
- symbols: its main definitions and exports by name (functions, classes, types, commands, sections), at most 12.
- notes: up to 4 things someone changing or using the file must know: invariants, pitfalls, unusual conventions, how
  it is tested or built. Empty when there is nothing beyond the obvious.
