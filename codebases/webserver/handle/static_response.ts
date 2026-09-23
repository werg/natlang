import type { Request, Parsed, Route, RouteDef, Entry, Session, Site, Review, Response } from "../types.js";
export default function static_response(acc: Site, route: Route): Response {
const body = acc.assets[route.asset]
if (body === undefined) return { status: 404, headers: { "Content-Type": "text/plain; charset=utf-8" }, body: "Not found" }
const type = route.asset.endsWith(".css") ? "text/css" : route.asset.endsWith(".txt") ? "text/plain" : "text/html"
return { status: 200, headers: { "Content-Type": type + "; charset=utf-8", "Cache-Control": "max-age=3600" }, body }
}
