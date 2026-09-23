import add_to_calendar from "./handle_mail/add_to_calendar";
import find_date from "./handle_mail/find_date";
import from_landlord from "./handle_mail/from_landlord";
---
description: Decide what to do with one email; dates from my landlord go into the calendar.
args:
  email: string
returns: Decision
effects: [calendar.add]
---
function handle_mail(email) -> Decision

  landlord = from_landlord(email)
  if not landlord:
      return { action: "archive", date: "" }
  date = find_date(email)                               # "" when the email names no date
  if date is "":
      return { action: "reply_later", date: "" }
  add_to_calendar(date, email)                          # effect
  return { action: "calendar", date }
