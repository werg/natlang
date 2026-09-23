export default function wrap_page(acc: Site, route: Route, content: string, session: Session): Response {
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
const safe = content.replace(/<\s*script[\s\S]*?<\s*\/\s*script\s*>/gi, "").replace(/\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "")
const nav = acc.routes.filter(r => r.kind === "page").map(r => `<a href="${r.path}">${esc(r.path === "/" ? "home" : r.path.slice(1))}</a>`).join(" · ")
const html = `<!doctype html><html><head><meta charset="utf-8"><title>${esc(acc.name)}</title><link rel="stylesheet" href="/style.css"></head>` +
             `<body><header><h1>${esc(acc.name)}</h1><nav>${nav}</nav></header><main>${safe}</main></body></html>`
return { status: 200, headers: { "Content-Type": "text/html; charset=utf-8", "Set-Cookie": `sid=${session.id}; Path=/; HttpOnly; SameSite=Lax` }, body: html }
}
