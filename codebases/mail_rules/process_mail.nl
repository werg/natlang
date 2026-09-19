---
description: Personal mail rules - what my landlord dates go into the calendar, the rest is sorted.
args:
  emails: Text[]
returns: MailReport
effects: [calendar.add]
types:
  Decision: '{ action: "calendar" | "reply_later" | "archive", date: Text }'
  MailReport: '{ calendar: Num, reply_later: Num, archived: Num }'
---
function process_mail(emails) -> MailReport

  decisions = for each e in emails: handle_mail(e)
  return tally_actions(decisions)                       # exact: how many of each action
