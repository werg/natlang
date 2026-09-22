import { highlight_file } from "./highlight/highlight_file";
---
description: Semantic syntax highlighting for a natlang code base - one HTML document per source file.
args:
  files: SourceFile[]
returns: Highlighted[]
types:
  SourceFile: '{ path: Text, text: Text }'
  Role: '"signature" | "call" | "call_each" | "repeat" | "condition" | "exact" | "prose_step" | "return" | "comment" | "blank" | "leaf_text"'
  Parts: '{ frontmatter: Text, lines: Text[], functions: Text[], is_code: Bool }'
  Highlighted: '{ path: Text, roles: Role[], html: Text }'
---
function highlight(files) -> Highlighted[]

  return for each f in files: highlight_file(f)
