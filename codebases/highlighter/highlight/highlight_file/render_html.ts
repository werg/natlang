import type { SourceFile, Role, Parts, Highlighted } from "../../types.js";
export default function render_html(file: SourceFile, parts: Parts, roles: Role[]): string {
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
const fns = new Set(parts.functions)
const KEY = new Set(["function", "for", "each", "in", "if", "else", "otherwise", "return", "repeat", "until", "at", "most", "times"])
function tokens(line) {
  const hash = line.search(/(^|\s)#/)
  const code = hash >= 0 ? line.slice(0, hash) : line
  const comment = hash >= 0 ? line.slice(hash) : ""
  const out = esc(code).replace(/`[^`]*`|"[^"]*"|\b[A-Za-z_]\w*\b|\b\d+(?:\.\d+)?\b/g, t =>
    t[0] === "`" ? `<span class="nl-path">${t}</span>` :
    t[0] === '"' ? `<span class="nl-str">${t}</span>` :
    /^\d/.test(t) ? `<span class="nl-num">${t}</span>` :
    fns.has(t) ? `<span class="nl-fn">${t}</span>` :
    KEY.has(t) ? `<span class="nl-kw">${t}</span>` : t)
  return out + (comment ? `<span class="nl-comment">${esc(comment)}</span>` : "")
}
const css = `body{font:14px/1.5 ui-monospace,monospace;background:#fbfaf7;color:#222;margin:2em}
.nl-front{color:#777}.nl-line{display:block;padding:0 .5em;border-left:3px solid transparent}
.nl-role-signature{font-weight:700}.nl-role-call,.nl-role-call_each{border-color:#2a7}.nl-role-repeat{border-color:#27a}
.nl-role-condition{border-color:#a72}.nl-role-exact{border-color:#a27;background:#fdf3f8}.nl-role-prose_step{border-color:#999;font-style:italic}
.nl-role-return{border-color:#555}.nl-fn{color:#0a6;font-weight:600}.nl-kw{color:#05a}.nl-path{color:#a50}.nl-str{color:#a22}
.nl-num{color:#70a}.nl-comment{color:#999}`
const front = parts.frontmatter ? `<pre class="nl-front">---\n${esc(parts.frontmatter)}\n---</pre>` : ""
const lines = parts.lines.map((l, i) =>
  `<span class="nl-line nl-role-${parts.is_code ? "code" : (roles[i] || "blank")}">${tokens(l) || " "}</span>`).join("\n")
return `<!doctype html><html><head><meta charset="utf-8"><title>${esc(file.path)}</title><style>${css}</style></head>` +
       `<body><h3>${esc(file.path)}</h3>${front}<pre class="nl-body">${lines}</pre></body></html>`
}
