export default function match_route(routes: RouteDef[], req: Parsed): Route {
const hit = routes.find(r => r.path === req.path)
if (!hit) return { kind: "not_found", name: req.path, purpose: "", asset: "" }
if (!hit.methods.includes(req.method)) return { kind: "method_not_allowed", name: hit.path, purpose: hit.methods.join(", "), asset: "" }
return { kind: hit.kind, name: hit.path, purpose: hit.purpose, asset: hit.asset || "" }
}
