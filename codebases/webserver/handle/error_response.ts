/*---
description: 404 or 405.
args:
  route: Route
returns: Response
---*/
const notAllowed = args.route.kind === "method_not_allowed"
return { status: notAllowed ? 405 : 404, headers: Object.assign({ "Content-Type": "text/html; charset=utf-8" }, notAllowed ? { "Allow": args.route.purpose } : {}),
         body: `<!doctype html><title>${notAllowed ? 405 : 404}</title><h1>${notAllowed ? "Method not allowed" : "Not found"}</h1><p><a href="/">home</a></p>` }
