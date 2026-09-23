import type { Request, Parsed, Route, RouteDef, Entry, Session, Site, Review, Response } from "../types.js";
import { effects as fx } from "natlang:runtime";

export default function respond(id: string, response: Response): boolean {
fx.http.respond(id, response)
return true
}
