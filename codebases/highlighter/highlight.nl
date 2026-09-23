---
description: Semantic syntax highlighting for a natlang code base - one HTML
  document per source file.
args:
  files: SourceFile[]
returns: Highlighted[]
---
function highlight(files) -> Highlighted[]

  return for each f in files: highlight_file(f)
