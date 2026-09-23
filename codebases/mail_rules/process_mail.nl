---
description: Personal mail rules - what my landlord dates go into the calendar,
  the rest is sorted.
args:
  emails: string[]
returns: MailReport
---
function process_mail(emails) -> MailReport

  decisions = for each e in emails: handle_mail(e)
  return tally_actions(decisions)                       # exact: how many of each action
