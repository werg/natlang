export default function session_for(acc: Site, req: Parsed): Session {
const old = acc.sessions[req.session_id]
if (old) return { ...old, visits: old.visits + 1 }
return { id: "s" + (Object.keys(acc.sessions).length + 1) + "x" + req.id.replace(/[^A-Za-z0-9]/g, ""), visits: 1, name: "" }
}
