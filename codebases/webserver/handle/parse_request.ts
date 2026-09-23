export default function parse_request(item: Request): Parsed {
const dec = s => { try { return decodeURIComponent(s.replace(/\+/g, " ")) } catch (e) { return s } }
const pairs = s => Object.fromEntries((s || "").split("&").filter(Boolean).map(p => { const i = p.indexOf("="); return i < 0 ? [dec(p), ""] : [dec(p.slice(0, i)), dec(p.slice(i + 1))] }))
const [path, qs] = item.path.split("?")
const cookie = Object.entries(item.headers).find(([k]) => lower(k) === "cookie")
const sid = cookie ? (cookie[1].match(/(?:^|;\s*)sid=([A-Za-z0-9_-]+)/) || [])[1] : ""
return { id: item.id, method: item.method.toUpperCase(), path: path.replace(/\/+$/, "") || "/", query: pairs(qs),
         form: item.method.toUpperCase() === "POST" ? pairs(item.body) : {}, session_id: sid || "" }
}
