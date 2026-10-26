---
description: Judge candidate identity edges as same, different, related or uncertain without suppressing conflict.
args: { dossier: LinkageDossier; change: LinkageChange; candidates: HistoricalCandidate[]; dates: DateInterval[]; places: HistoricalPlace[] }
returns: HistoricalIdentityFinding[]
---
Call `compareDateIntervals(candidates, dates)` using interval overlap, not point-date equality. Assess relationship and location evidence alongside attributes. Return different when timing or relationship evidence rules out identity; related when evidence links distinct people; same only when positive identity evidence supports it; otherwise uncertain. Record conflicts and source IDs explicitly.
