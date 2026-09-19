/*---
description: Put page content into the site layout, with headers and the session cookie.
args:
  acc: Site
  route: Route
  content: Text
  session: Session
returns: Response
---*/
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
const safe = args.content.replace(/<\s*script[\s\S]*?<\s*\/\s*script\s*>/gi, "").replace(/\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "")
const nav = args.acc.routes.filter(r => r.kind === "page").map(r => `<a href="${r.path}">${esc(r.path === "/" ? "home" : r.path.slice(1))}</a>`).join(" · ")
const html = `<!doctype html><html><head><meta charset="utf-8"><title>${esc(args.acc.name)}</title><link rel="stylesheet" href="/style.css"></head>` +
             `<body><header><h1>${esc(args.acc.name)}</h1><nav>${nav}</nav></header><main>${safe}</main></body></html>`
return { status: 200, headers: { "Content-Type": "text/html; charset=utf-8", "Set-Cookie": `sid=${args.session.id}; Path=/; HttpOnly; SameSite=Lax` }, body: html }
