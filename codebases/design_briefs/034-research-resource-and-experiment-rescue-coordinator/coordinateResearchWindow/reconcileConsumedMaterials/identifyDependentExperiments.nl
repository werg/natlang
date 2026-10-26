---
args: { trace: MaterialTrace, snapshot: ResearchOperationsSnapshot }
returns: DependencyImpact
---
Follow explicit protocol and sample-dependency edges from each affected experiment. Return only downstream experiments whose required input was consumed, delayed, or cancelled. A shared project label alone does not establish dependency.
