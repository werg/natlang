/*---
description: The legal action with this code; anything else becomes decline.
args:
  wish: string
  legal: Action[]
returns: Action
---*/
return args.legal.find(a => a.code === args.wish.trim()) || args.legal.find(a => a.code === "decline")
