# 074 — Volunteer Deployment Coordinator

## Distinct algorithm: meet a mission's qualification coverage with consenting people and supervised roles

A mission is a set of capability requirements with time, place and access constraints. The coordinator builds a coverage relation from each requirement to volunteers whose qualification is verified and current for the assignment window. Headcount is irrelevant when a required skill is missing. Expired qualifications remove only the roles that depend on them; unrelated assignments remain valid. Team construction then checks availability and explicit consent for the actual work mode, especially when a remote assignment changes to on-site.

The assignment graph preserves supervisory edges: a trainee can fill a role only when a qualified supervisor is scheduled in the same accepted assignment window. Private access instructions are projected to the smallest role-specific briefing and only for the period the assignment is active. When someone withdraws, the coordinator recomputes the capability coverage and marks uncovered requirements, rather than merely substituting the same number of volunteers.

The resumable mission board retains qualification evidence versions, consent scope and mode, supervision pairings, briefing disclosure windows, assignments, withdrawals and completion receipts. Language proposes a feasible team and briefing; the host validates credential freshness, permissions and scheduling conflicts.
