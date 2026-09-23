import type { Request, Parsed, Route, RouteDef, Entry, Session, Site, Review, Response } from "../types.js";
export default function log_request(site: Site, req: Parsed, session: Session, response: Response): Site {
const line = `${req.method} ${req.path} -> ${response.status} (${response.body.length} bytes, session ${session.id}, visit ${session.visits})`
return { ...site, sessions: { ...site.sessions, [session.id]: session }, log: site.log.concat([line]).slice(-200) }
}
