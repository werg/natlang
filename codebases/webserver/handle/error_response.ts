import type { Request, Parsed, Route, RouteDef, Entry, Session, Site, Review, Response } from "../types.js";
export default function error_response(route: Route): Response {
const notAllowed = route.kind === "method_not_allowed"
return { status: notAllowed ? 405 : 404, headers: Object.assign({ "Content-Type": "text/html; charset=utf-8" }, notAllowed ? { "Allow": route.purpose } : {}),
         body: `<!doctype html><title>${notAllowed ? 405 : 404}</title><h1>${notAllowed ? "Method not allowed" : "Not found"}</h1><p><a href="/">home</a></p>` }
}
