---
description: Retention of open incidents. Choose which open incidents leave the open list when there are more than the list holds.
args:
  open: Incident[]
  settings: LogSettings
returns: string[]
---
open is the list of incidents still open, and settings.max_open is how many the list holds. Choose the IDs of the
incidents that leave the list so that settings.max_open of them stay.

1. leaving = the number of incidents in open less settings.max_open. When it is 0 or less, answer an empty list.
2. Rank the incidents by how much a person watching the logs still needs them. An incident ranks higher when it has an
   alert_key, a severity of "high" or "critical", or a last_seen that is recent; and lower when its count is small and
   its last_seen is old.
3. Answer the IDs of the `leaving` lowest-ranked incidents, each ID once, taken from open.
