import line_role from "./highlight_file/line_role";
import render_html from "./highlight_file/render_html";
import split_source from "./highlight_file/split_source";
---
description: Highlight one source file - split it exactly, judge the role of every line, render HTML exactly.
args:
  file: SourceFile
returns: Highlighted
---
function highlight_file(file) -> Highlighted

  parts = split_source(file)                                 # exact: frontmatter, body lines, names of functions it can call
  if parts.is_code:                                          # a .ts function: nothing to judge
      roles = []
  else:
      roles = for each l in parts.lines: line_role(l, parts.functions)
  html = render_html(file, parts, roles)
  return { path: file.path, roles, html }
