export default function split_source(file: SourceFile): Parts {
const text = file.text
const isCode = file.path.endsWith(".ts")
const importBlock = isCode ? "" : (text.match(/^(?:import\s+[^\n]+\n)+\n?/)?.[0] || "")
const moduleText = importBlock ? text.slice(importBlock.length) : text
const m = isCode ? moduleText.match(/^\s*\/\*---\n([\s\S]*?)\n---\*\/\n?([\s\S]*)$/) : moduleText.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
const front = (importBlock.trimEnd() + (importBlock && m ? "\n" : "") + (m ? m[1] : ""))
const body = (m ? m[2] : moduleText).replace(/\n+$/, "")
const names = new Set()
for (const entry of importBlock.matchAll(/import\s+([A-Za-z_$][\w$]*)\s+from\s+["'][^"']+["']/g)) names.add(entry[1])
for (const c of body.matchAll(/\b([A-Za-z_]\w*)\(/g)) if (!["function", "if", "for", "while"].includes(c[1])) names.add(c[1])
return { frontmatter: front, lines: body === "" ? [] : body.split("\n"), functions: [...names], is_code: isCode }
}
