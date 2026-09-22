import { highlight_file } from "./highlight/highlight_file";
---
description: Semantic syntax highlighting for a natlang code base - one HTML document per source file.
args:
  files: SourceFile[]
returns: Highlighted[]
types:
  SourceFile: '{ path: string, text: string }'
  Role: '"signature" | "call" | "call_each" | "repeat" | "condition" | "exact" | "prose_step" | "return" | "comment" | "blank" | "leaf_text"'
  Parts: '{ frontmatter: string, lines: string[], functions: string[], is_code: boolean }'
  Highlighted: '{ path: string, roles: Role[], html: string }'
---
function highlight(files) -> Highlighted[]

  return for each f in files: highlight_file(f)
