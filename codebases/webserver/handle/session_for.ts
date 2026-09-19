/*---
description: The visitor's session - the existing one with one more visit, or a new one.
args:
  acc: Site
  req: Parsed
returns: Session
---*/
const old = args.acc.sessions[args.req.session_id]
if (old) return { ...old, visits: old.visits + 1 }
return { id: "s" + (Object.keys(args.acc.sessions).length + 1) + "x" + args.req.id.replace(/[^A-Za-z0-9]/g, ""), visits: 1, name: "" }
