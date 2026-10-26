---
args: { arrangements: TemporaryArrangement[] }
returns: string[]
---
Transfer each unexpired temporary arrangement as a dated follow-up obligation with the named owner. If an expiry has passed or owner is missing, preserve the obligation and flag it for accountable review rather than closing it at handover.
