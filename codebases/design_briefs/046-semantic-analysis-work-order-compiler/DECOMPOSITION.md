# Semantic analysis work-order compiler

The compiler formalizes population, predicates, time, grouping, and output grain before looking for available fields. A missing temporal window remains a question when alternate interpretations change the result. Fields must match the concept's documented meaning; inaccessible records leave the analysis universe before semantic classification and are reported without exposing their values.

The typed work graph carries source lineage and missing-data rules across joins and aggregation. Each join is checked against the declared output grain so a one-to-many relationship cannot silently inflate counts. Run audits reconcile every row disposition and pin semantic judgments to their rubric version. A rubric change creates a targeted new run while preserving the old one; a grouping-only change can reuse qualified row judgments. Numerical results are stated only when materialized evidence is supplied.

`reconcileAnalysisMaterialization.nl` binds the executed artifact to its graph, schema snapshot, row coverage, output grain, rubric lineage, and authorized audience. An artifact that fails any check stays hidden and cannot replace the current run; rubric revisions preserve old judgments under their original version until affected rows are reclassified.
