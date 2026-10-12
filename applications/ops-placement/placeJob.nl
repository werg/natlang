---
description: Decides where one natlang job runs (DGX, Pop, a CoreWeave GPU sandbox, the DGX queue, or the owner's decision) from the job's needs and the current facts of the fleet.
args:
  job: JobNeeds
  fleet: FleetFacts
returns: PlacementDecision
---
job describes one job: what it needs and who owns its lineage. fleet describes the machines now. DGX costs nothing extra but its memory is shared with the running training; a CoreWeave sandbox has one GPU of fleet.coreweave_gpu_memory_gb GB, costs money, and cannot reach the teacher on DGX. Go through the steps in order and stop at the first that applies.

1. If job.lineage_owner is "pop": where "pop", reason "Pop owns this lineage".
2. If job.moves_active_lineage is true: where "ask", reason "moving the active lineage of a line is the owner's decision" followed by where it would go.
3. If job.needs_teacher is true: when job.gpu_memory_gb is at most fleet.dgx_headroom_gb, where "dgx", otherwise where "queue-dgx"; reason "needs the teacher on DGX".
4. If job.gpu_memory_gb is at most fleet.dgx_headroom_gb and fleet.dgx_waiting_ahead is 0: where "dgx", reason "fits the DGX headroom now at no extra cost".
5. If fleet.coreweave_ready is true, job.gpu_memory_gb is at most fleet.coreweave_gpu_memory_gb minus fleet.coreweave_fit_margin_gb, fleet.coreweave_active is less than fleet.coreweave_max_parallel, and either fleet.coreweave_budget_left_usd is null or job.coreweave_cost_usd is a number at most fleet.coreweave_budget_left_usd: where "coreweave", reason "DGX cannot take it now" followed by the DGX headroom and queue, and the cost when it is a number.
6. Otherwise: where "queue-dgx", reason naming the first thing that keeps it off CoreWeave: not ready, too large for one GPU, the parallel limit, or the budget.
Return { where, reason }.
