/*---
description: Serve a stored asset.
args:
  acc: Site
  route: Route
returns: Response
---*/
const body = args.acc.assets[args.route.asset]
if (body === undefined) return { status: 404, headers: { "Content-Type": "text/plain; charset=utf-8" }, body: "Not found" }
const type = args.route.asset.endsWith(".css") ? "text/css" : args.route.asset.endsWith(".txt") ? "text/plain" : "text/html"
return { status: 200, headers: { "Content-Type": type + "; charset=utf-8", "Cache-Control": "max-age=3600" }, body }
