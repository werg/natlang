/*---
description: Send the response for a request.
args:
  id: string
  response: Response
returns: boolean
effects: [http.respond]
---*/
fx.http.respond(args.id, args.response)
return true
