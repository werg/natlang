/*---
description: How many decisions of each kind.
args:
  decisions: Decision[]
returns: MailReport
---*/
const n = a => args.decisions.filter(d => d.action === a).length
return { calendar: n("calendar"), reply_later: n("reply_later"), archived: n("archive") }
