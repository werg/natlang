/*---
description: Send the response for a request.
args:
  id: Text
  response: Response
returns: Bool
effects: [http.respond]
---*/
fx.http.respond(args.id, args.response)
return true
