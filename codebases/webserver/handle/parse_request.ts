/*---
description: Query string, form fields and session cookie of a request.
args:
  item: Request
returns: Parsed
---*/
const dec = s => { try { return decodeURIComponent(s.replace(/\+/g, " ")) } catch (e) { return s } }
const pairs = s => Object.fromEntries((s || "").split("&").filter(Boolean).map(p => { const i = p.indexOf("="); return i < 0 ? [dec(p), ""] : [dec(p.slice(0, i)), dec(p.slice(i + 1))] }))
const [path, qs] = args.item.path.split("?")
const cookie = Object.entries(args.item.headers).find(([k]) => lower(k) === "cookie")
const sid = cookie ? (cookie[1].match(/(?:^|;\s*)sid=([A-Za-z0-9_-]+)/) || [])[1] : ""
return { id: args.item.id, method: args.item.method.toUpperCase(), path: path.replace(/\/+$/, "") || "/", query: pairs(qs),
         form: args.item.method.toUpperCase() === "POST" ? pairs(args.item.body) : {}, session_id: sid || "" }
