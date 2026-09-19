/*---
description: Which route a request is for.
args:
  routes: RouteDef[]
  req: Parsed
returns: Route
---*/
const hit = args.routes.find(r => r.path === args.req.path)
if (!hit) return { kind: "not_found", name: args.req.path, purpose: "", asset: "" }
if (!hit.methods.includes(args.req.method)) return { kind: "method_not_allowed", name: hit.path, purpose: hit.methods.join(", "), asset: "" }
return { kind: hit.kind, name: hit.path, purpose: hit.purpose, asset: hit.asset || "" }
