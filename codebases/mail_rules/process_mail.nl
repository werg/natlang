import { handle_mail } from "./process_mail/handle_mail";
import { tally_actions } from "./process_mail/tally_actions";
---
description: Personal mail rules - what my landlord dates go into the calendar, the rest is sorted.
args:
  emails: string[]
returns: MailReport
effects: [calendar.add]
types:
  Decision: '{ action: "calendar" | "reply_later" | "archive", date: string }'
  MailReport: '{ calendar: number, reply_later: number, archived: number }'
---
function process_mail(emails) -> MailReport

  decisions = for each e in emails: handle_mail(e)
  return tally_actions(decisions)                       # exact: how many of each action
