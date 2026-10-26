---
args: { join: AnalysisJoin, outputGrain: string }
returns: JoinGrainCheck
---
Compare the declared one-to-one, many-to-one, or one-to-many relationship with the requested output grain. Where a one-to-many join would repeat an entity, require an explicit deduplication or aggregation rule before the join is counted. Return the exact key and fanout relationship that changes the number of output rows.
