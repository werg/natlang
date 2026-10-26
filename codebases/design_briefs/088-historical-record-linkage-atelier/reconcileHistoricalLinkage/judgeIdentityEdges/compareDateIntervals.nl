---
description: Compare historical date intervals for possible overlap without collapsing approximate dates to point values.
args: { candidates: HistoricalCandidate[]; dates: DateInterval[] }
returns: string[]
---
Use supplied earliest and latest bounds in the same calendar basis. Intervals that cannot overlap are chronologically incompatible; overlapping intervals permit but do not prove identity. Preserve unresolved calendar conversions as unknown. Never infer identity from chronology compatibility alone.
