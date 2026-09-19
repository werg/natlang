/*---
description: Remember the session and append one line to the log.
args:
  site: Site
  req: Parsed
  session: Session
  response: Response
returns: Site
---*/
const line = `${args.req.method} ${args.req.path} -> ${args.response.status} (${args.response.body.length} bytes, session ${args.session.id}, visit ${args.session.visits})`
return { ...args.site, sessions: { ...args.site.sessions, [args.session.id]: args.session }, log: args.site.log.concat([line]).slice(-200) }
