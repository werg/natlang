/*---
description: Split a source file into frontmatter and body lines; list the functions it names in `uses` and calls.
args:
  file: SourceFile
returns: Parts
---*/
const text = args.file.text
const isCode = args.file.path.endsWith(".ts")
const m = isCode ? text.match(/^\s*\/\*---\n([\s\S]*?)\n---\*\/\n?([\s\S]*)$/) : text.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
const front = m ? m[1] : ""
const body = (m ? m[2] : text).replace(/\n+$/, "")
const names = new Set()
const uses = front.match(/^uses:\n((?:[ \t]+.*\n?)*)/m)
if (uses) for (const l of uses[1].split("\n")) { const k = l.match(/^\s+([A-Za-z_]\w*):/); if (k) names.add(k[1]) }
for (const c of body.matchAll(/\b([A-Za-z_]\w*)\(/g)) if (!["function", "if", "for", "while"].includes(c[1])) names.add(c[1])
return { frontmatter: front, lines: body === "" ? [] : body.split("\n"), functions: [...names], is_code: isCode }
