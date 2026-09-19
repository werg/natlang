/*---
description: Add an accepted guestbook entry to the site; a refused one changes nothing.
args:
  acc: Site
  review: Review
returns: Site
---*/
if (!args.review.accept || !args.review.message.trim()) return args.acc
const entry = { author: args.review.author.trim().slice(0, 60) || "anonymous", message: args.review.message.trim().slice(0, 500) }
return { ...args.acc, entries: args.acc.entries.concat([entry]).slice(-50) }
