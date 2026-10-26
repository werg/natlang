# 077 — Care Transition Logistics Coordinator

## Distinct algorithm: unlock a care-transition task graph from confirmed prerequisites and handoff receipts

A transition plan is a directed graph of logistical tasks, each requiring named prerequisites, a responsible role and completion evidence. The coordinator checks actual delivery and acceptance receipts before unlocking downstream work. Equipment booked at a destination is not equipment delivered; a caregiver named in a plan is not a caregiver who accepted responsibility. The dependency graph makes those gaps block only their dependent steps.

Schedule changes propagate through exact time windows to downstream appointments, accessible transport and destination access. A proposed substitution that touches a clinical requirement is routed to the clinician and remains blocked until review; the coordinator handles logistics, not care decisions. The receiving organization must acknowledge the plan before the handoff is ready. Recovery after a delay records the last completed task, unresolved prerequisites and affected appointment windows so the transition can resume without claiming an unobserved event.
