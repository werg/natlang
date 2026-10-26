---
args: { assignments: RepairerMatch[], windows: SharedToolWindow[] }
returns: ToolReservationReview
---
Reserve a shared tool only for the exact task and time window when no other assignment occupies it. Return overlapping reservation IDs and tasks without needed tools; do not assume a tool becomes free after an unverified completion.
