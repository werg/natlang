---
args: { graph: WorkGraphProposal, snapshot: Snapshot }
returns: EvidencePlan
---
1. Specify source records and fields for each task, exact search partitions, context windows, and missing coverage.
2. Specify the records, fields, context windows, and negative examples needed by each node. Use exact filters before semantic retrieval. Track searched partitions and missing coverage; retrieval absence never proves nonexistence outside the searched universe.
3. Return an explicit artifact; preserve unknowns and source coverage. This procedure proposes no production write.
